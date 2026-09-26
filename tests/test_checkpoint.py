import json

import numpy as np
import torch

from agent import Agent
from helper import load_plot_data, save_plot_data


def test_agent_checkpoint_restores_training_state(tmp_path):
    agent = Agent()
    agent.n_games = 7
    agent.record = 3
    agent.plot_scores = [1, 2, 3]
    agent.plot_mean_scores = [1.0, 1.5, 2.0]
    agent.total_score = 6
    state = np.zeros(11, dtype=int)
    agent.remember(state, [1, 0, 0], 0, state, False)

    checkpoint_path = tmp_path / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    restored = Agent()
    restored.load_checkpoint(checkpoint_path)

    assert restored.n_games == 7
    assert restored.record == 3
    assert restored.plot_scores == [1, 2, 3]
    assert restored.plot_mean_scores == [1.0, 1.5, 2.0]
    assert restored.total_score == 6
    assert len(restored.memory) == 1
    assert torch.equal(restored.model.linear1.weight, agent.model.linear1.weight)


def test_plot_data_round_trips_as_json(tmp_path):
    path = tmp_path / "training_stats.json"
    save_plot_data(path, [1, 2], [1.0, 1.5])

    assert load_plot_data(path) == {
        "scores": [1, 2],
        "mean_scores": [1.0, 1.5],
    }
    assert json.loads(path.read_text())
