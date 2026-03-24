import json
import matplotlib.pyplot as plt

with open("src/data/models/train_log.json") as f:
    log = json.load(f)

episodes = [e["episode"] for e in log]
rewards  = [e["total_reward"] for e in log]
losses   = [e["mean_loss"] for e in log if e["mean_loss"] is not None]
epsilons = [e["epsilon"] for e in log]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(episodes, rewards)
axes[0,0].set_title("Reward per Episode")

axes[0,1].plot(range(len(losses)), losses)
axes[0,1].set_title("Mean Loss per Episode")

axes[1,0].plot(episodes, epsilons)
axes[1,0].set_title("Epsilon Decay")

window = max(1, len(rewards) // 20)
smoothed = [sum(rewards[max(0,i-window):i+1]) / len(rewards[max(0,i-window):i+1])
            for i in range(len(rewards))]
axes[1,1].plot(episodes, smoothed)
axes[1,1].set_title(f"Reward (smoothed, window={window})")

plt.tight_layout()
plt.savefig("src/data/graphs/training_curves.png", dpi=150)
plt.show()