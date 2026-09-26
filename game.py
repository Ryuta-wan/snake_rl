import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()


def _pick_font(size, bold=False):
    """เลือกฟอนต์แบบมีตัวสำรอง (เครื่องที่ไม่มี Segoe UI จะ fallback ได้)"""
    for name in ('segoe ui', 'tahoma', 'dejavusans', 'arial'):
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            continue
    return pygame.font.Font(None, size)


font = _pick_font(25)  # เก็บไว้เพื่อความเข้ากันได้กับโค้ดเดิม

# กำหนดทิศทางแบบ absolute (ใช้อ้างอิงตอนเช็คชนกำแพง/วาดภาพ)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# ค่าคงที่ของเกม
BLOCK_SIZE = 20
SPEED = 10

# ---------------------------------------------------------------- สีของ UI
# Modern dark palette (ธีมเดียวกับ snake-rl-pro)
BG = (15, 18, 28)
GRID = (28, 34, 49)
SNAKE_HEAD = (82, 220, 166)
SNAKE_BODY = (42, 164, 132)
FOOD = (255, 103, 122)
TEXT = (235, 241, 255)
MUTED = (145, 157, 182)
EYE = (255, 255, 255)
PUPIL = (18, 26, 38)

# ชื่อสีเดิม เก็บไว้ไม่ให้โค้ด/เอกสารส่วนอื่นพัง
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # ตั้งค่าหน้าจอ pygame
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.font = _pick_font(20)
        self.small_font = _pick_font(15)
        self.reset()

    def reset(self):
        # เริ่มเกมใหม่: งูเริ่มตรงกลาง หันขวา ยาว 3 ท่อน
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # สุ่มตำแหน่งอาหาร ต้องไม่ทับตัวงู
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. เช็ค event ปิดหน้าต่าง (ให้ปิดเกมได้ปกติ)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. ขยับหัวงูตาม action ที่ agent เลือก
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. เช็คว่าจบเกมหรือยัง (ชนกำแพง/ชนตัวเอง หรือเดินนานเกินไป)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. เช็คว่ากินอาหารหรือไม่
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # ถ้าไม่กิน ตัดหางออก (งูไม่ยาวขึ้น)

        # 5. วาดเฟรมใหม่
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # ชนกำแพง
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # ชนตัวเอง
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self, draw_hud=True):
        # พื้นหลังเข้ม + เส้นกริดจาง ๆ
        self.display.fill(BG)
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID, (0, y), (self.w, y))

        # ตัวงู: สี่เหลี่ยมขอบมนเล็กน้อย หัวสีอ่อนกว่าลำตัว
        for index, point in enumerate(self.snake):
            color = SNAKE_HEAD if index == 0 else SNAKE_BODY
            rect = pygame.Rect(point.x + 1, point.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            pygame.draw.rect(self.display, color, rect, border_radius=4)
        self._draw_eyes()

        # อาหาร
        food_rect = pygame.Rect(self.food.x + 3, self.food.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6)
        pygame.draw.rect(self.display, FOOD, food_rect, border_radius=8)

        # HUD (ปิดได้ด้วย draw_hud=False เผื่อสคริปต์อื่นอยากวาด HUD เอง)
        if draw_hud:
            label = self.font.render(f'SCORE  {self.score}', True, TEXT)
            hint = self.small_font.render('SNAKE RL  •  Deep Q-Network', True, MUTED)
            self.display.blit(label, (14, 12))
            self.display.blit(hint, (14, 38))
            pygame.display.flip()

    def _draw_eyes(self):
        """วาดลูกตาขาว 2 จุดบนหัวงู หันไปตามทิศที่งูเดิน"""
        head = self.snake[0]
        cx = head.x + BLOCK_SIZE / 2
        cy = head.y + BLOCK_SIZE / 2

        # forward = ทิศที่มอง, side = แกนตั้งฉาก (ใช้แยกตาซ้าย/ขวา)
        forward = {
            Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
            Direction.DOWN: (0, 1), Direction.UP: (0, -1),
        }[self.direction]
        side = (-forward[1], forward[0])

        ahead = BLOCK_SIZE * 0.20   # เลื่อนตาไปข้างหน้า
        apart = BLOCK_SIZE * 0.24   # ระยะห่างระหว่างตาสองข้าง
        size = max(3, round(BLOCK_SIZE * 0.30))          # ขนาดตา (สี่เหลี่ยมจิ๋ว)
        pupil_size = max(1, round(size * 0.45))

        for sign in (-1, 1):
            ex = cx + forward[0] * ahead + side[0] * apart * sign
            ey = cy + forward[1] * ahead + side[1] * apart * sign
            eye_rect = pygame.Rect(0, 0, size, size)
            eye_rect.center = (int(ex), int(ey))
            pygame.draw.rect(self.display, EYE, eye_rect, border_radius=1)

            pupil_rect = pygame.Rect(0, 0, pupil_size, pupil_size)
            pupil_rect.center = (
                int(ex + forward[0] * size * 0.22),
                int(ey + forward[1] * size * 0.22),
            )
            pygame.draw.rect(self.display, PUPIL, pupil_rect)

    def _move(self, action):
        # action = [1,0,0] ตรงไป, [0,1,0] เลี้ยวขวา, [0,0,1] เลี้ยวซ้าย
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]          # ตรงไป: ทิศทางเดิม
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # เลี้ยวขวา: หมุนตามเข็มนาฬิกา
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # เลี้ยวซ้าย: หมุนทวนเข็มนาฬิกา

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


# ทดสอบเล่นเองด้วยคีย์บอร์ด (ยังไม่มี RL ตรงนี้)
if __name__ == '__main__':
    game = SnakeGameAI()

    while True:
        action = [1, 0, 0]  # default: เดินตรง
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                idx = clock_wise.index(game.direction)
                if event.key == pygame.K_RIGHT:
                    target = Direction.RIGHT
                elif event.key == pygame.K_LEFT:
                    target = Direction.LEFT
                elif event.key == pygame.K_UP:
                    target = Direction.UP
                elif event.key == pygame.K_DOWN:
                    target = Direction.DOWN
                else:
                    target = game.direction
                target_idx = clock_wise.index(target)
                if target_idx == idx:
                    action = [1, 0, 0]
                elif target_idx == (idx + 1) % 4:
                    action = [0, 1, 0]
                elif target_idx == (idx - 1) % 4:
                    action = [0, 0, 1]

        reward, game_over, score = game.play_step(action)
        if game_over:
            print('Final Score:', score)
            pygame.time.wait(2000)
            break

    pygame.quit()