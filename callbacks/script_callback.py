import os
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class ScriptCallback(Callback):
    def __init__(self, script_path=None, script_args=''):
        super().__init__()
        self.script_path = script_path
        self.script_args = script_args

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, **kwargs) -> None:
        try:
            if self.script_path is not None:
                print(f'INFO: try to run {self.script_path}')
                subprocess.Popen(['bash', self.script_path, self.script_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                # subprocess.Popen(['bash', self.script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) # deamonize the process so that it keeps running even if the main process is killed
        except Exception as e:
            print(f'ERROR when trying to run {self.script_path}: {e}')
        return super().on_train_epoch_start(trainer, pl_module, **kwargs)

class ParentNodeOnlyScriptCallback(ScriptCallback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, **kwargs) -> None:
        if pl_module.global_rank == 0:
            if pl_module.local_rank == 0:
                print(f'INFO: Running {self.script_path} on global rank {pl_module.global_rank}')
                super().on_train_epoch_start(trainer, pl_module, **kwargs)
        else:
            print(f'INFO: Skip running {self.script_path} on global rank {pl_module.global_rank}')

class ChildrenNodeOnlyScriptCallback(ScriptCallback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, **kwargs) -> None:
        if pl_module.global_rank != 0:
            if pl_module.local_rank == 0:
                print(f'INFO: Running {self.script_path} on global rank {pl_module.global_rank}')
                super().on_train_epoch_start(trainer, pl_module, **kwargs)
        else:
            print(f'INFO: Skip running {self.script_path} on global rank {pl_module.global_rank}')