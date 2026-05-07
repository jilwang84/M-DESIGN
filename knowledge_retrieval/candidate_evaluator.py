"""Candidate model evaluation through the released GraphGym runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


TASK_TO_GRAPHGYM = {
    "node_classification": "node",
    "link_prediction": "link_pred",
    "graph_classification": "graph",
}

TASK_CONFIG = {
    "node_classification": "improved_v2.yaml",
    "link_prediction": "improved_v2.yaml",
    "graph_classification": "improved_v2_graph.yaml",
}

GRAPHGYM_STAGE_ALIASES = {
    "ppr_01": "ppr_0.1",
}

CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    """Runtime options for evaluating a candidate architecture."""

    graphgym_root: Path
    output_root: Path
    gpu_id: int = 0
    repeat: int = 3
    max_epoch: int | None = None
    python_executable: str = sys.executable
    timeout: int | None = None


class CandidateEvaluator:
    """Train and evaluate candidate architectures when no DB record is available."""

    def __init__(self, task: str, config: CandidateEvaluationConfig):
        if task not in TASK_TO_GRAPHGYM:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.config = config

    def evaluate(self, dataset: str, model: Mapping[str, object]) -> tuple[float, float]:
        """Run GraphGym and return the aggregated test metric and std."""
        run_id = self._run_id(dataset, model)
        run_dir = self.config.output_root / self.task / dataset / run_id
        cfg_path = run_dir / "candidate.yaml"
        graphgym_out_dir = run_dir / "candidate"
        if (graphgym_out_dir / "agg" / "test" / "best.json").exists():
            return self._read_aggregated_score(graphgym_out_dir)

        run_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_graphgym_config(dataset, model, run_dir)
        with cfg_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        graphgym_run_dir = self.config.graphgym_root / "run"
        command = [
            self.config.python_executable,
            "main_pyg.py",
            "--cfg",
            cfg_path.as_posix(),
            "--repeat",
            str(self.config.repeat),
            "--gpu_id",
            str(self.config.gpu_id),
        ]
        env = os.environ.copy()
        pythonpath_parts = [str(self.config.graphgym_root), env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
        subprocess.run(
            command,
            cwd=graphgym_run_dir,
            env=env,
            check=True,
            timeout=self.config.timeout,
        )
        return self._read_aggregated_score(graphgym_out_dir)

    def _build_graphgym_config(
        self,
        dataset: str,
        model: Mapping[str, object],
        run_dir: Path,
    ) -> dict[str, object]:
        config_path = self._graphgym_config_path()
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        config["out_dir"] = str(run_dir)
        config["dataset"]["format"] = "PyG"
        config["dataset"]["name"] = dataset
        config["dataset"]["task"] = TASK_TO_GRAPHGYM[self.task]

        if self.task == "node_classification":
            config["gnn"]["neigh"] = model["neigh"]
        elif self.task == "link_prediction":
            config["model"]["edge_decoding"] = model["decode"]
        elif self.task == "graph_classification":
            config["model"]["graph_pooling"] = model["decode"]

        config["gnn"]["norm_mode"] = model["norm"]
        config["gnn"]["agg"] = model["agg"]
        config["gnn"]["self_msg"] = model["comb"]
        config["gnn"]["layers_mp"] = int(model["l_mp"])
        config["gnn"]["stage_type"] = GRAPHGYM_STAGE_ALIASES.get(model["stage"], model["stage"])
        if self.config.max_epoch is not None:
            config["optim"]["max_epoch"] = self.config.max_epoch
        return config

    @staticmethod
    def _read_aggregated_score(run_dir: Path) -> tuple[float, float]:
        best_path = run_dir / "agg" / "test" / "best.json"
        if not best_path.exists():
            raise FileNotFoundError(f"GraphGym did not write {best_path}")
        with best_path.open("r", encoding="utf-8") as file:
            stats = json.loads(file.readline())
        metric = "accuracy" if "accuracy" in stats else "auc"
        return float(stats[metric]), float(stats.get(f"{metric}_std", 0.0))

    def _graphgym_config_path(self) -> Path:
        return (
            self.config.graphgym_root
            / "run"
            / "configs"
            / "improved"
            / TASK_CONFIG[self.task]
        )

    def _graphgym_config_digest(self) -> str:
        config_bytes = self._graphgym_config_path().read_bytes()
        return hashlib.sha1(config_bytes).hexdigest()

    def _run_id(self, dataset: str, model: Mapping[str, object]) -> str:
        payload = json.dumps(
            {
                "cache_schema": CACHE_SCHEMA_VERSION,
                "task": self.task,
                "dataset": dataset,
                "model": dict(model),
                "repeat": self.config.repeat,
                "max_epoch": self.config.max_epoch,
                "graphgym_config_sha1": self._graphgym_config_digest(),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
