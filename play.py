"""
play.py — โหลด agent ที่เทรนไว้แล้วมาเล่นโชว์วนไปเรื่อย ๆ (ไม่เทรน ไม่เซฟทับ)

ใช้งาน:
    python play.py                  # เล่นวนเรื่อย ๆ ที่ความเร็ว 20 FPS
    python play.py --speed 60       # เร็วขึ้น
    python play.py --games 5        # เล่นแค่ 5 เกมแล้วจบ
    python play.py --epsilon 0.02   # ใส่การสุ่มนิดหน่อย กันงูติดลูป
    python play.py --weights model/model_final.pth

กด ESC หรือปิดหน้าต่างเพื่อออก / กด SPACE เพื่อพัก-เล่นต่อ
"""

import argparse
from pathlib import Path

import pygame
import torch

import game as game_mod
from game import BLOCK_SIZE, SnakeGameAI
from model import Linear_QNet

CHECKPOINT_PATH = Path('model') / 'checkpoint.pth'
FINAL_WEIGHTS = Path('model') / 'model_final.pth'
LEGACY_WEIGHTS = Path('model') / 'model.pth'


def load_policy(weights=None):
    """โหลดน้ำหนักมาใส่โมเดล แล้วคืน (model, n_games, record, ที่มา)"""
    model = Linear_QNet(11, 256, 3)

    # 1. ถ้าระบุไฟล์มาเอง ใช้ไฟล์นั้นก่อน
    if weights is not None:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f'Weights not found: {path}')
        blob = torch.load(path, map_location='cpu', weights_only=False)
        state = blob.get('model_state_dict', blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state)
        model.eval()
        n_games = blob.get('n_games', 0) if isinstance(blob, dict) else 0
        record = blob.get('record', 0) if isinstance(blob, dict) else 0
        return model, n_games, record, str(path)

    # 2. ไม่ได้ระบุ: เอา checkpoint ก่อน (มีทั้ง n_games / record ให้โชว์)
    for path in (CHECKPOINT_PATH, CHECKPOINT_PATH.with_suffix('.pth.bak')):
        if path.exists():
            try:
                ck = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(ck['model_state_dict'])
                model.eval()
                return model, ck.get('n_games', 0), ck.get('record', 0), str(path)
            except Exception as error:
                print(f'skip {path}: {error}')

    # 3. สุดท้ายลองไฟล์ weight ล้วน
    for path in (FINAL_WEIGHTS, LEGACY_WEIGHTS):
        if path.exists():
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            model.eval()
            return model, 0, 0, str(path)

    raise FileNotFoundError(
        'ไม่เจอโมเดลเลย — รัน `python agent.py --resume` ให้เทรนจนได้ checkpoint ก่อนนะ'
    )


def get_state(game):
    """state 11 ค่า — ต้องเรียงเหมือนตอนเทรนเป๊ะ ๆ ไม่งั้น weight ใช้ไม่ได้"""
    from game import Direction, Point

    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    return [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),
        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),
        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),
        dir_l, dir_r, dir_u, dir_d,
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
    ]


class DemoHUD:
    """แปะสถิติการเล่นทับบนหน้าจอเกม (ไม่ยุ่งกับ game.py)"""

    def __init__(self, game, source, trained_games, record):
        self.game = game
        self.source = Path(source).name
        self.trained_games = trained_games
        self.train_record = record
        self.played = 0
        self.best = 0
        self.total = 0

    def finish_game(self, score):
        self.played += 1
        self.total += score
        self.best = max(self.best, score)

    def draw(self, paused=False):
        g = self.game
        mean = self.total / self.played if self.played else 0.0

        lines = [
            (f'SCORE  {g.score}', g.font, game_mod.TEXT),
            (f'best {self.best}   mean {mean:.1f}   games {self.played}',
             g.small_font, game_mod.MUTED),
        ]
        y = 12
        for text, font, color in lines:
            g.display.blit(font.render(text, True, color), (14, y))
            y += 26

        footer = (f'{self.source}  •  trained {self.trained_games} games  '
                  f'•  record {self.train_record}')
        surface = g.small_font.render(footer, True, game_mod.MUTED)
        g.display.blit(surface, (14, g.h - surface.get_height() - 10))

        if paused:
            msg = g.font.render('PAUSED  —  press SPACE', True, game_mod.TEXT)
            rect = msg.get_rect(center=(g.w // 2, g.h // 2))
            pad = 14
            box = rect.inflate(pad * 2, pad * 2)
            overlay = pygame.Surface(box.size, pygame.SRCALPHA)
            overlay.fill((*game_mod.BG, 225))
            g.display.blit(overlay, box.topleft)
            pygame.draw.rect(g.display, game_mod.SNAKE_BODY, box, width=1, border_radius=8)
            g.display.blit(msg, rect)

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description='เล่นโชว์ด้วยโมเดลที่เทรนไว้แล้ว')
    parser.add_argument('--weights', default=None,
                        help='ไฟล์ checkpoint หรือ weight (ไม่ระบุ = model/checkpoint.pth)')
    parser.add_argument('--speed', type=int, default=20, help='FPS (ค่าเริ่มต้น 20)')
    parser.add_argument('--games', type=int, default=5, help='เล่นกี่เกมแล้วหยุด (0 = วนไม่หยุด)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='โอกาสสุ่ม action 0-1 ช่วยแก้ตอนงูเดินวนติดลูป')
    args = parser.parse_args()

    model, trained_games, record, source = load_policy(args.weights)
    print(f'loaded {source}  (trained {trained_games} games, record {record})')
    print('ESC / ปิดหน้าต่าง = ออก   |   SPACE = พัก-เล่นต่อ')

    game = SnakeGameAI()
    pygame.display.set_caption('Snake RL — Demo (no training)')
    hud = DemoHUD(game, source, trained_games, record)

    import random
    clock = pygame.time.Clock()
    running = True
    paused = False

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                hud.draw(paused=True)
                clock.tick(15)
                continue

            state = get_state(game)
            if args.epsilon > 0 and random.random() < args.epsilon:
                move_index = random.randrange(3)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float))
                move_index = int(torch.argmax(q_values))
            action = [0, 0, 0]
            action[move_index] = 1

            # เดินเกม 1 step เอง (ไม่เรียก play_step เพราะมันวาด HUD ของตัวเอง
            # และ tick ที่ SPEED=2000 ซึ่งเร็วเกินจะดูรู้เรื่อง)
            game.frame_iteration += 1
            game._move(action)
            game.snake.insert(0, game.head)

            done = game.is_collision() or game.frame_iteration > 100 * len(game.snake)
            if not done:
                if game.head == game.food:
                    game.score += 1
                    game._place_food()
                else:
                    game.snake.pop()

            game._update_ui(draw_hud=False)   # วาด HUD เองด้านล่าง
            hud.draw()
            clock.tick(args.speed)

            if done:
                score = game.score
                hud.finish_game(score)
                mean = hud.total / hud.played
                print(f'game {hud.played}  score {score}  best {hud.best}  mean {mean:.2f}')
                pygame.time.wait(600)
                game.reset()
                if args.games and hud.played >= args.games:
                    running = False
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        if hud.played:
            print(f'\nplayed {hud.played} games | best {hud.best} | '
                  f'mean {hud.total / hud.played:.2f}')
        pygame.quit()


if __name__ == '__main__':
    main()
