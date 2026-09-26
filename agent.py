import torch
import random
import numpy as np
import os
import pickle
from pathlib import Path
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot, save_plot_data, load_plot_data

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_GAMES = 25_000   # เทรนถึงเกมที่เท่านี้แล้วหยุด (None = ไม่จำกัด)


CHECKPOINT_PATH = Path('model') / 'checkpoint.pth'
PLOT_DATA_PATH = Path('model') / 'training_stats.json'
PLOT_IMAGE_PATH = Path('model') / 'training_plot.png'


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0          # ควบคุมการสุ่ม (exploration)
        self.gamma = 0.9          # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # เก็บประสบการณ์ ถ้าเต็มจะลบอันเก่าสุดออกอัตโนมัติ
        self.model = Linear_QNet(11, 256, 3)     # input=11, hidden=256, output=3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.record = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0

    def get_state(self, game):
        head = game.snake[0]
        # จุดรอบหัวงู 4 ทิศ ใช้เช็คอันตราย
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
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

            # ทิศทางปัจจุบัน
            dir_l, dir_r, dir_u, dir_d,

            # ตำแหน่งอาหารเทียบกับหัวงู
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # เก็บประสบการณ์ 1 ครั้งลง memory (เป็น tuple)
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # สุ่มหยิบ batch จาก memory มาเทรน (เรียกตอนจบเกมแต่ละรอบ)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # เทรนทันทีจาก step ล่าสุดแค่ครั้งเดียว (เรียกทุก step)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # epsilon-greedy: ยิ่งเล่นไปหลายเกม epsilon ยิ่งลด -> สุ่มน้อยลง ใช้โมเดลมากขึ้น
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)              # explore: สุ่มเดิน
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)           # exploit: ใช้โมเดลทำนาย
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_checkpoint(self, path=CHECKPOINT_PATH):
        """Save everything needed to continue training, not only weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
            'record': self.record,
            'memory': list(self.memory),
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
            'total_score': self.total_score,
        }
        temp_path = path.with_suffix(path.suffix + '.tmp')
        backup_path = path.with_suffix(path.suffix + '.bak')
        torch.save(payload, temp_path)
        with temp_path.open('r+b') as checkpoint_file:
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        if path.exists():
            os.replace(path, backup_path)
        os.replace(temp_path, path)

    def load_checkpoint(self, path=CHECKPOINT_PATH):
        """Restore a checkpoint created by ``save_checkpoint``."""
        path = Path(path)
        candidates = [path, path.with_suffix(path.suffix + '.bak')]
        checkpoint = None
        errors = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
                print(f'Loaded checkpoint: {candidate}')
                break
            except (RuntimeError, EOFError, OSError, ValueError, pickle.UnpicklingError) as error:
                errors.append(f'{candidate}: {error}')
        if checkpoint is None:
            legacy_model = Path('model') / 'model.pth'
            if legacy_model.exists():
                self.model.load(legacy_model)
                stats = load_plot_data(PLOT_DATA_PATH)
                self.plot_scores = stats.get('scores', [])
                self.plot_mean_scores = stats.get('mean_scores', [])
                self.n_games = len(self.plot_scores)
                self.record = max(self.plot_scores, default=0)
                self.total_score = sum(self.plot_scores)
                print(f'Checkpoint was corrupted; recovered model weights from {legacy_model}')
                return
            details = '\n'.join(errors) or f'No checkpoint found at {path}'
            raise RuntimeError(
                'No valid checkpoint or legacy model could be loaded.\n' + details +
                '\nRun `python agent.py` to start a new training session.'
            )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint.get('n_games', 0)
        self.record = checkpoint.get('record', 0)
        self.memory = deque(checkpoint.get('memory', []), maxlen=MAX_MEMORY)
        self.plot_scores = checkpoint.get('plot_scores', [])
        self.plot_mean_scores = checkpoint.get('plot_mean_scores', [])
        self.total_score = checkpoint.get('total_score', sum(self.plot_scores))

        # Keep the standalone stats file useful even if it was deleted.
        save_plot_data(PLOT_DATA_PATH, self.plot_scores, self.plot_mean_scores)


def train(resume=False, target_games=TARGET_GAMES):
    agent = Agent()
    if resume and CHECKPOINT_PATH.exists():
        agent.load_checkpoint()
    elif resume:
        legacy_model = Path('model') / 'model.pth'
        if legacy_model.exists():
            agent.model.load(legacy_model)
            print(f'Loaded legacy weights from {legacy_model}; optimizer and history start fresh.')
        else:
            raise FileNotFoundError(
                f'Checkpoint not found: {CHECKPOINT_PATH} (and no legacy model at {legacy_model})'
            )

    if target_games is not None and agent.n_games >= target_games:
        print(f'Already at {agent.n_games} games (target {target_games}). Nothing to train.')
        return agent

    if target_games is not None:
        print(f'Training from game {agent.n_games} up to {target_games}...')

    game = SnakeGameAI()

    while True:
        # 1. หา state ปัจจุบัน
        state_old = agent.get_state(game)

        # 2. เลือก action
        final_move = agent.get_action(state_old)

        # 3. เล่นเกม 1 step ตาม action นั้น
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. เทรนแบบ short memory (ทุก step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. เก็บลง memory ไว้เทรนซ้ำทีหลัง
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # จบเกมแล้ว: reset, เทรนแบบ long memory, เก็บสถิติ
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)

            agent.plot_scores.append(score)
            agent.total_score += score
            mean_score = agent.total_score / agent.n_games
            agent.plot_mean_scores.append(mean_score)
            save_plot_data(PLOT_DATA_PATH, agent.plot_scores, agent.plot_mean_scores)
            plot(agent.plot_scores, agent.plot_mean_scores, PLOT_IMAGE_PATH)
            agent.save_checkpoint()

            # ถึงเป้าหมายแล้ว: เซฟทุกอย่างเรียบร้อยแล้วค่อยหยุด
            if target_games is not None and agent.n_games >= target_games:
                agent.model.save('model_final.pth')
                print(f'\nReached {agent.n_games} games. Training stopped.')
                print(f'  record     : {agent.record}')
                print(f'  mean score : {mean_score:.2f}')
                print(f'  checkpoint : {CHECKPOINT_PATH}')
                print(f'  weights    : model/model_final.pth')
                return agent


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='load model, optimizer, replay memory and plot history')
    parser.add_argument('--games', type=int, default=TARGET_GAMES,
                        help=f'stop after reaching this many games (default {TARGET_GAMES}); '
                             'use 0 for unlimited')
    args = parser.parse_args()
    train(resume=args.resume, target_games=args.games or None)