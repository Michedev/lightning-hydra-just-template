import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch


class GradientMonitorCallback(Callback):
    """Callback to monitor gradient norms and learning rate during training."""
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called after loss.backward() and before optimizers.step()."""
        if trainer.global_step % self.log_every_n_steps == 0:
            # Calculate gradient norms
            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Log gradient norm
            pl_module.log("monitor/grad_norm", total_norm, on_step=True, on_epoch=False)
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        """Called after training batch ends."""
        if trainer.global_step % self.log_every_n_steps == 0:
            # Log learning rate
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            pl_module.log("monitor/learning_rate", current_lr, on_step=True, on_epoch=False)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of training epoch."""
        print(f"\n[Epoch {trainer.current_epoch}] Training completed")
        print(f"Global step: {trainer.global_step}")
