import matplotlib.pyplot as plt
from IPython import display
import json
from pathlib import Path

plt.ion()  # เปิดโหมด interactive ให้กราฟอัปเดตสดได้ระหว่างรัน


def save_plot_data(path, scores, mean_scores):
    """Persist the values behind the live plot so training can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'scores': list(scores),
        'mean_scores': list(mean_scores),
    }, indent=2), encoding='utf-8')


def load_plot_data(path):
    path = Path(path)
    if not path.exists():
        return {'scores': [], 'mean_scores': []}
    return json.loads(path.read_text(encoding='utf-8'))


def plot(scores, mean_scores, save_path=None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()  # เคลียร์กราฟเก่าออกก่อนวาดใหม่ทุกครั้ง

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    plt.ylim(ymin=0)

    # แปะตัวเลขคะแนนล่าสุดไว้ที่จุดสุดท้ายของกราฟ
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)