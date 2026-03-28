"""Trainer — runs the training loop over train days, then evaluates on test days."""

import json
import os
from typing import Any

from modelling.agent import Agent


class Trainer:

    def __init__(
        self,
        agent:       Agent,
        split_path:  str,
        flows_dir:   str,
        output_dir:  str  = "src/data/models",
        n_epochs:    int  = 3,
        save_every:  int  = 10,
        log_every:   int  = 5,
    ):
        self.agent      = agent
        self.output_dir = output_dir
        self.n_epochs   = n_epochs
        self.save_every = save_every
        self.log_every  = log_every

        os.makedirs(output_dir, exist_ok=True)
        self._checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        with open(split_path) as f:
            self._split = json.load(f)

        self._train_days = self._split["train"]
        self._test_days  = self._split["test"]
        self._flows_dir  = flows_dir
        self._train_log: list[dict] = []
        self._test_log:  list[dict] = []

    def run(self) -> dict[str, Any]:
        """Full training loop + final evaluation. Returns train_log and test_log."""
        print(f"\nStarting training")
        print(f"  Train days : {len(self._train_days)}")
        print(f"  Test days  : {len(self._test_days)}")
        print(f"  Epochs     : {self.n_epochs}")
        print(f"  Output     : {self.output_dir}\n")

        episode = 0
        for epoch in range(1, self.n_epochs + 1):
            print(f"--- Epoch {epoch}/{self.n_epochs} ---")
            for day_id in self._train_days:
                episode += 1
                metrics = self._run_episode(day_id, train=True)
                metrics["episode"] = episode
                metrics["epoch"]   = epoch
                metrics["day_id"]  = day_id
                self._train_log.append(metrics)

                if episode % self.log_every == 0:
                    self._log(episode, metrics, prefix="Train")

                if episode % self.save_every == 0:
                    path = os.path.join(
                        self._checkpoint_dir, f"checkpoint_ep{episode:04d}.pt"
                    )
                    self.agent.save(path)
                    print(f"  Checkpoint saved -> {path}")

        print(f"\n--- Evaluating on {len(self._test_days)} test days ---")
        self.agent.set_eval_mode()
        for test_episode, day_id in enumerate(self._test_days, start=1):
            metrics = self._run_episode(day_id, train=False)
            metrics["day_id"] = day_id
            self._test_log.append(metrics)
            self._log(test_episode, metrics, prefix="Test ")

        final_path = os.path.join(self.output_dir, "final_model.pt")
        self.agent.save(final_path)
        print(f"\nFinal model saved -> {final_path}")

        self._save_logs()
        self._print_summary()

        return {"train_log": self._train_log, "test_log": self._test_log}

    def _run_episode(self, day_id: int, train: bool) -> dict[str, Any]:
        """Run one episode (one day) in train or eval mode."""
        route_file = os.path.join(
            self._flows_dir, f"flows_day_{day_id:02d}.rou.xml"
        )
        if not os.path.exists(route_file):
            raise FileNotFoundError(
                f"Route file not found: {route_file}\n"
                f"Run BuildRoute.py first to generate flow files."
            )

        if train:
            self.agent.set_train_mode()
        else:
            self.agent.set_eval_mode()

        obs  = self.agent.start_episode(route_file)
        done = False
        while not done:
            obs, rewards, done, loss = self.agent.step(obs)

        return self.agent.end_episode()

    def _log(self, index: int, metrics: dict, prefix: str = "") -> None:
        eps = metrics.get("epsilon")
        eps_str = f"  ε={eps:.3f}" if eps is not None else ""
        loss = metrics.get("mean_loss")
        loss_str = f"  loss={loss:.4f}" if loss is not None else ""
        lr = metrics.get("learning_rate")
        lr_str = f"  lr={lr:.6f}" if lr is not None else ""
        print(
            f"  {prefix} ep={index:4d}"
            f"  reward={metrics['total_reward']:8.1f}"
            f"  steps={metrics['steps']:5d}"
            f"{loss_str}{eps_str}{lr_str}"
        )

    def _save_logs(self) -> None:
        train_path = os.path.join(self.output_dir, "train_log.json")
        test_path  = os.path.join(self.output_dir, "test_log.json")
        with open(train_path, "w") as f:
            json.dump(self._train_log, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(self._test_log, f, indent=2)
        print(f"Logs saved -> {train_path}")
        print(f"           -> {test_path}")

    def _print_summary(self) -> None:
        if self._train_log:
            mean_reward = sum(m["total_reward"] for m in self._train_log) / len(self._train_log)
            print(f"\nTrain mean reward : {mean_reward:.2f}")
        if self._test_log:
            mean_reward = sum(m["total_reward"] for m in self._test_log) / len(self._test_log)
            print(f"Test  mean reward : {mean_reward:.2f}")
