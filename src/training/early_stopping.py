"""Early stopping on validation loss (lower is better)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EarlyStopping:
    """Stop when validation loss has not improved by more than ``min_delta`` for ``patience`` epochs.

    An epoch counts as an improvement when ``loss < best_loss - min_delta``.
    """

    patience: int = 4
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    best_epoch: Optional[int] = None
    epochs_without_improvement: int = 0
    should_stop: bool = False
    reason: str = ""
    history: list = field(default_factory=list)

    def step(self, epoch: int, val_loss: float) -> bool:
        """Record one epoch's validation loss. Returns True if this epoch is the new best."""
        if val_loss != val_loss:  # NaN
            self.should_stop = True
            self.reason = f"validation loss became NaN at epoch {epoch}"
            self.history.append({"epoch": epoch, "val_loss": val_loss, "improved": False})
            return False
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                self.reason = (f"no improvement > {self.min_delta} in validation loss for {self.patience} epochs "
                               f"(best {self.best_loss:.5f} at epoch {self.best_epoch})")
        self.history.append({"epoch": epoch, "val_loss": val_loss, "improved": improved})
        return improved

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
