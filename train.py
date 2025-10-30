import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import  DataLoader
import torch
from dipper.paths import ROOT
import os
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

# Register OmegaConf resolvers
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
OmegaConf.register_new_resolver("pow", lambda x, y: x ** y)
OmegaConf.register_new_resolver("int", lambda x: int(x))
OmegaConf.register_new_resolver("float", lambda x: float(x))
OmegaConf.register_new_resolver('get_fname', lambda x: x.split('.')[-2])


@hydra.main(ROOT / 'config', 'train')
def main(cfg: DictConfig):

    print('Config:', cfg, sep='\n')
    
    # Get the number of GPUs allocated by Slurm
    if "SLURM_JOB_GPUS" in os.environ:
        # SLURM_JOB_GPUS contains comma-separated GPU IDs
        gpus = len(os.environ["SLURM_JOB_GPUS"].split(","))
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        # Default to using all available GPUs
        gpus = torch.cuda.device_count()
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)


    train_dloader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    test_dloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    model = hydra.utils.instantiate(cfg.model)

    trainer: pl.Trainer  = hydra.utils.instantiate(cfg.trainer)
    if cfg.validate_only:
        assert cfg.ckpt_path is not None
        from torch.utils.data import Subset
        train_dloader_limited = DataLoader(Subset(train_dataset, list(range(cfg.batch_size))), cfg.batch_size)
        trainer._loggers = [CSVLogger('.', 'validation_output')]
        output = trainer.validate(model, train_dloader_limited, ckpt_path=ROOT / cfg.ckpt_path)
        return output

    ckpt_path = (ROOT / cfg.ckpt_path)  if  cfg.ckpt_path is not None  else None
    trainer.fit(model, train_dloader, test_dloader, ckpt_path=ckpt_path)
    


if __name__ == '__main__':
    main()