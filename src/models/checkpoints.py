"""Checkpoints pattern for model resilience during training."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import lightgbm as lgb  # type: ignore
import pandas as pd


class ModelCheckpoint:
    """
    Design Pattern: Checkpoints
    - Saves model state at regular intervals
    - Allows training to resume after interruption
    - Enables recovery from failures
    """

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        save_freq: int = 10,
        keep_best: bool = True,
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N iterations
            keep_best: Keep best checkpoint based on validation metric
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.keep_best = keep_best
        self.max_checkpoints = max_checkpoints
        self.best_score = None
        self.best_iteration = None
        self.checkpoint_count = 0

    def save_checkpoint(
        self,
        model: lgb.Booster,
        iteration: int,
        metrics: Dict[str, float],
        label_mapping: Dict[str, Any],
        is_best: bool = False,
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: LightGBM model
            iteration: Current training iteration
            metrics: Current metrics
            label_mapping: Label mapping dictionary
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_iter_{iteration}.txt"
        if is_best:
            checkpoint_name = "checkpoint_best.txt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save model
        model.save_model(str(checkpoint_path))

        # Save metadata
        metadata = {
            "iteration": iteration,
            "metrics": metrics,
            "label_mapping": label_mapping,
            "is_best": is_best,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        metadata_path = self.checkpoint_dir / f"{checkpoint_path.stem}_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        self.checkpoint_count += 1
        print(f"Checkpoint saved: {checkpoint_path} (iteration {iteration})")

        # Clean old checkpoints if needed
        if self.checkpoint_count > self.max_checkpoints:
            self._clean_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None, load_best: bool = False
    ) -> tuple:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to specific checkpoint (optional)
            load_best: Load best checkpoint if True

        Returns:
            Tuple of (model, metadata)
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.txt"
        elif checkpoint_path is None:
            # Load latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_iter_*.txt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        model = lgb.Booster(model_file=str(checkpoint_path))

        # Load metadata
        metadata_path = self.checkpoint_dir / f"{checkpoint_path.stem}_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
        else:
            metadata = {"iteration": 0, "metrics": {}, "label_mapping": {}}

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Iteration: {metadata.get('iteration', 'unknown')}")
        print(f"  Metrics: {metadata.get('metrics', {})}")

        return model, metadata

    def _clean_old_checkpoints(self):
        """Remove old checkpoints, keeping only the best and recent ones."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_iter_*.txt"))

        # Always keep best checkpoint
        best_path = self.checkpoint_dir / "checkpoint_best.txt"

        # Keep only recent checkpoints
        checkpoints_to_keep = checkpoints[-self.max_checkpoints :]

        for checkpoint in checkpoints:
            if checkpoint not in checkpoints_to_keep and checkpoint != best_path:
                checkpoint.unlink()
                metadata_path = (
                    self.checkpoint_dir / f"{checkpoint.stem}_metadata.joblib"
                )
                if metadata_path.exists():
                    metadata_path.unlink()
                print(f"Removed old checkpoint: {checkpoint}")


def create_checkpoint_callback(
    checkpoint_manager: ModelCheckpoint, validation_metric: str = "val_accuracy"
):
    """
    Create LightGBM callback for checkpointing.

    Args:
        checkpoint_manager: ModelCheckpoint instance
        validation_metric: Metric to use for best model selection

    Returns:
        Callback function
    """

    def callback(env):
        """LightGBM callback function."""
        iteration = env.iteration

        # Save checkpoint at regular intervals
        if iteration > 0 and iteration % checkpoint_manager.save_freq == 0:
            # Get current metrics
            metrics = {}
            if env.evaluation_result_list:
                for eval_result in env.evaluation_result_list:
                    metric_name = eval_result[0]
                    metric_value = eval_result[1]
                    metrics[metric_name] = metric_value

            # Check if this is the best model
            if validation_metric in metrics:
                current_score = metrics[validation_metric]
                if (
                    checkpoint_manager.best_score is None
                    or current_score > checkpoint_manager.best_score
                ):
                    checkpoint_manager.best_score = current_score
                    checkpoint_manager.best_iteration = iteration

            # Save checkpoint (metadata will be saved separately)
            # Note: In actual implementation, we'd need to pass model and label_mapping
            # This is a simplified version for the callback

    return callback
