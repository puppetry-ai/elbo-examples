from typing import Any

import torch
import os
from datetime import datetime
from data.midi_data_module import MidiDataModule
import torch.nn.functional as F
import wandb

from data.mnist_data_module import MNISTDataModule
from utils.cuda_utils import get_device


class BaseModel(torch.nn.Module):
    """
    Base model class that will be inherited by all model types
    """

    def __init__(self, lr=1e-3,
                 data_dir=os.path.expanduser("~/midi_features/"),
                 output_dir=os.path.expanduser("~/model-archive-note-seq/"),
                 num_gpus=1,
                 batch_size=3000,
                 sample_output_step=200,
                 save_checkpoint_every=1000,
                 emit_tensorboard_scalars=True,
                 use_mnist_dms=False,
                 *args: Any, **kwargs: Any):
        super(BaseModel, self).__init__(*args, **kwargs)
        now = datetime.now()
        now_str = now.strftime(format="%d-%h-%y")
        self._data_dir = data_dir
        self._output_dir = os.path.join(output_dir, now_str)
        os.makedirs(self._output_dir, exist_ok=True)
        self._lr = lr
        self._use_mnist_dms = use_mnist_dms
        self._dms = None
        self._model_prefix = "base-model"
        self._num_gpus = num_gpus
        self._sample_output_step = sample_output_step
        self._save_checkpoint_every = save_checkpoint_every
        self._emit_tensorboard_scalars = emit_tensorboard_scalars
        self._batch_size = batch_size
        self._data_mean = None
        self._data_std = None

    def setup(self):
        if self._use_mnist_dms:
            self._dms = MNISTDataModule(self._data_dir,
                                        batch_size=self._batch_size)
        else:
            self._dms = MidiDataModule(self._data_dir,
                                       batch_size=self._batch_size)

        self._dms.setup()
        if not self._use_mnist_dms:
            self._data_mean, self._data_std = self._dms.get_mean_and_std()

    @staticmethod
    def sample(mean, var):
        # Set all small values to epsilon
        std = torch.sqrt(var)
        std = F.softplus(std) + 1e-8
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def _kl_simple(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    @staticmethod
    def enable_debugging():
        torch.autograd.set_detect_anomaly(True)

    def loss_function(self, x_hat_bp, x_hat, x, x_control, qm, qv):
        raise NotImplementedError(f"Please implement loss_function()")

    def step(self, batch, batch_idx):
        raise NotImplementedError(f"Please implement step()")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def sample_output(self, epoch):
        raise NotImplementedError(f"Please implement sample_output()")

    def fit(self, epoch, optimizer):
        device = get_device()
        self.to(device)
        self.train()
        batch_train_loss = 0
        batch_kl_loss = 0.0
        batch_recon_loss = 0.0
        batch_pitches_loss = 0.0
        batch_velocity_loss = 0.0
        batch_instruments_loss = 0.0
        batch_program_loss = 0.0
        batch_start_time_loss = 0.0
        batch_end_time_loss = 0.0
        batch_control_loss = 0.0

        for batch_idx, batch in enumerate(self._dms.train_dataloader()):
            optimizer.zero_grad()
            loss, kl, recon_loss, pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss = self.step(batch, batch_idx)
            loss.backward()
            optimizer.step()

            batch_train_loss += loss.detach().item()
            batch_kl_loss += kl.detach().item()
            batch_recon_loss += recon_loss.detach().item()

            batch_pitches_loss += pitches_loss.detach().item()
            batch_velocity_loss += velocity_loss.detach().item()
            batch_instruments_loss += instruments_loss.detach().item()
            batch_program_loss += program_loss.detach().item()
            batch_start_time_loss += start_times_loss.detach().item()
            batch_end_time_loss += end_times_loss.detach().item()
            batch_control_loss += control_loss.detach().item()

        loss = batch_train_loss / len(self._dms.train_dataloader().dataset)
        kl_loss = batch_kl_loss / len(self._dms.train_dataloader().dataset)
        recon_loss = batch_recon_loss / len(self._dms.train_dataloader().dataset)
        pitches_loss = batch_pitches_loss/ len(self._dms.train_dataloader().dataset)
        velocity_loss= batch_velocity_loss/ len(self._dms.train_dataloader().dataset)
        instrument_loss = batch_instruments_loss/ len(self._dms.train_dataloader().dataset)
        program_loss = batch_program_loss/ len(self._dms.train_dataloader().dataset)
        start_times_loss = batch_start_time_loss/ len(self._dms.train_dataloader().dataset)
        end_times_loss = batch_end_time_loss/ len(self._dms.train_dataloader().dataset)
        control_loss = batch_control_loss/ len(self._dms.train_dataloader().dataset)

        wandb.log({'loss': loss})
        wandb.log({'kl_loss': kl_loss})
        wandb.log({'recon_loss': recon_loss})
        wandb.log({'pitches_ce_loss': pitches_loss})
        wandb.log({'velocity_ce_loss': velocity_loss})
        wandb.log({'instruments_ce_loss': instrument_loss})
        wandb.log({'program_ce_loss': program_loss})
        wandb.log({'start_time_l2_loss': start_times_loss})
        wandb.log({'duration_l2_loss': end_times_loss})
        wandb.log({'control_l2_loss': control_loss})
        print(f'====> Train Loss = {loss} KL = {kl_loss} Recon = {recon_loss} Control Loss={control_loss} Epoch = {epoch}')

    def save(self, epoch):
        model_save_path = os.path.join(self._output_dir, f"{self._model_prefix}-epoch-{epoch}-{wandb.run.name}.checkpoint")
        print(f"Saving model to --> {model_save_path}")
        torch.save(self.state_dict(), model_save_path)

    def test(self):
        self.eval()
        device = get_device()
        batch_test_loss = 0
        batch_kl_loss = 0.0
        batch_recon_loss = 0.0
        batch_pitches_loss = 0.0
        batch_velocity_loss = 0.0
        batch_instruments_loss = 0.0
        batch_program_loss = 0.0
        batch_start_time_loss = 0.0
        batch_end_time_loss = 0.0
        batch_control_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._dms.test_dataloader()):
                if self._use_mnist_dms:
                    batch = batch.reshape(-1, 28, 28)

                batch = batch.to(device)
                loss, kl, recon_loss, pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss = self.step(batch, batch_idx)
                batch_test_loss += loss.detach().item()
                batch_kl_loss += kl.detach().item()

                batch_recon_loss += recon_loss.detach().item()
                batch_pitches_loss += pitches_loss.detach().item()
                batch_velocity_loss += velocity_loss.detach().item()
                batch_instruments_loss += instruments_loss.detach().item()
                batch_program_loss += program_loss.detach().item()
                batch_start_time_loss += start_times_loss.detach().item()
                batch_end_time_loss += end_times_loss.detach().item()
                batch_control_loss += control_loss.detach().item()

        loss = batch_test_loss / len(self._dms.test_dataloader().dataset)
        kl_loss = batch_kl_loss / len(self._dms.test_dataloader().dataset)
        recon_loss = batch_recon_loss / len(self._dms.test_dataloader().dataset)
        pitches_loss = batch_pitches_loss/ len(self._dms.test_dataloader().dataset)
        velocity_loss= batch_velocity_loss/ len(self._dms.test_dataloader().dataset)
        instrument_loss = batch_instruments_loss/ len(self._dms.test_dataloader().dataset)
        program_loss = batch_program_loss/ len(self._dms.test_dataloader().dataset)
        start_times_loss = batch_start_time_loss/ len(self._dms.test_dataloader().dataset)
        end_times_loss = batch_end_time_loss/ len(self._dms.test_dataloader().dataset)
        control_loss = batch_control_loss/ len(self._dms.test_dataloader().dataset)

        wandb.log({'test_loss': loss})
        wandb.log({'test_kl_loss': kl_loss})
        wandb.log({'test_recon_loss': recon_loss})
        wandb.log({'test_pitches_ce_loss': pitches_loss})
        wandb.log({'test_velocity_ce_loss': velocity_loss})
        wandb.log({'test_instruments_ce_loss': instrument_loss})
        wandb.log({'test_program_ce_loss': program_loss})
        wandb.log({'test_start_time_l2_loss': start_times_loss})
        wandb.log({'test_duration_l2_loss': end_times_loss})
        wandb.log({'test_control_l2_loss': control_loss})
        print(f'====> Test Loss = {loss} KL = {kl_loss} Recon = {recon_loss} Control Loss = {control_loss}')


    @property
    def lr(self):
        return self._lr
