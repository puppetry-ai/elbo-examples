import os
import os.path
from datetime import datetime
from typing import Any
from typing import Optional

import elbo.elbo
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
import wandb
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class MNISTTensorDataSet(Dataset):
    def __init__(self, tensors, batch_size):
        self.batched_tensors = tensors

    def __getitem__(self, index):
        return self.batched_tensors[index]

    def __len__(self):
        return len(self.batched_tensors)


class MNISTDataModule(pl.LightningDataModule):

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(f"No Predict data loader for MNIST")

    def __init__(self,
                 data_dir,
                 batch_size=100,
                 shuffle=True,
                 data_shape=(28, 28),
                 ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataloader = None
        self._test_dataloader = None
        self._data_shape = data_shape
        self._num_workers = 0
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        device = get_device()
        train_x = np.array([np.array(x) for x, _ in datasets.MNIST(self._data_dir, train=True, download=True)])
        train_y = np.array([np.array(y) for _, y in datasets.MNIST(self._data_dir, train=True, download=True)])
        train_x = train_x.astype(np.float32) / 255.0
        train_x = [torch.tensor(x).to(device) for x in train_x]
        train_data_set = MNISTTensorDataSet(train_x, batch_size=self._batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_data_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        test_x = np.array([np.array(x) for x, _ in datasets.MNIST(self._data_dir, train=False, download=True)])
        test_y = np.array([np.array(y) for _, y in datasets.MNIST(self._data_dir, train=False, download=True)])
        test_x = test_x.astype(np.float32) / 255.0
        test_x = [torch.tensor(x).to(device) for x in test_x]
        test_y = test_y.astype(np.float32)
        test_data_set = MNISTTensorDataSet(test_x, batch_size=self._batch_size)
        test_loader = torch.utils.data.DataLoader(
            test_data_set,
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers)

        self._train_dataloader = train_loader
        self._test_dataloader = test_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(f"No VAL data loader for MNIST")

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


class BaseModel(torch.nn.Module, elbo.elbo.ElboModel):
    """
    Base model class that will be inherited by all model types
    """

    def get_artifacts_directory(self):
        return self._output_dir

    def save_state(self):
        self.save()

    def load_state(self, state_dir):
        pass

    def __init__(self, lr=1e-3,
                 data_dir=os.path.expanduser("data"),
                 output_dir=os.path.expanduser("artifacts"),
                 num_gpus=1,
                 batch_size=3000,
                 sample_output_step=200,
                 save_checkpoint_every=1000,
                 emit_tensorboard_scalars=True,
                 use_mnist_dms=False,
                 *args: Any, **kwargs: Any):
        super(BaseModel, self).__init__(*args, **kwargs)
        now = datetime.now()
        self._data_dir = data_dir
        self._output_dir = output_dir
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
        self._dms = MNISTDataModule(self._data_dir,
                                    batch_size=self._batch_size)
        self._dms.setup()

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

        for batch_idx, batch in enumerate(self._dms.train_dataloader()):
            optimizer.zero_grad()
            loss, kl, recon_loss, = self.step(batch, batch_idx)
            loss.backward()
            optimizer.step()

            batch_train_loss += loss.detach().item()
            batch_kl_loss += kl.detach().item()
            batch_recon_loss += recon_loss.detach().item()

        loss = batch_train_loss / len(self._dms.train_dataloader().dataset)
        kl_loss = batch_kl_loss / len(self._dms.train_dataloader().dataset)
        recon_loss = batch_recon_loss / len(self._dms.train_dataloader().dataset)

        wandb.log({'loss': loss})
        wandb.log({'kl_loss': kl_loss})
        wandb.log({'recon_loss': recon_loss})
        print(
            f'====> Train Loss = {loss} KL = {kl_loss} Recon = {recon_loss}  Epoch = {epoch}')

    def save(self):
        model_save_path = os.path.join(self._output_dir,
                                       f"{self._model_prefix}-{wandb.run.name}.checkpoint")
        print(f"Saving model to --> {model_save_path}")
        torch.save(self.state_dict(), model_save_path)

    def test(self):
        self.eval()
        device = get_device()
        batch_test_loss = 0
        batch_kl_loss = 0.0
        batch_recon_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._dms.test_dataloader()):
                batch = batch.reshape(-1, 28, 28)

                batch = batch.to(device)
                loss, kl, recon_loss, = self.step(batch, batch_idx)
                batch_test_loss += loss.detach().item()
                batch_kl_loss += kl.detach().item()

                batch_recon_loss += recon_loss.detach().item()

        loss = batch_test_loss / len(self._dms.test_dataloader().dataset)
        kl_loss = batch_kl_loss / len(self._dms.test_dataloader().dataset)
        recon_loss = batch_recon_loss / len(self._dms.test_dataloader().dataset)

        wandb.log({'test_loss': loss})
        wandb.log({'test_kl_loss': kl_loss})
        wandb.log({'test_recon_loss': recon_loss})
        print(f'====> Test Loss = {loss} KL = {kl_loss} Recon = {recon_loss}')

    @property
    def lr(self):
        return self._lr


class ExtractLSTMOutput(nn.Module):
    def __init__(self, extract_out=True):
        super(ExtractLSTMOutput, self).__init__()
        self._extract_out = extract_out

    def forward(self, x):
        out, hidden = x
        if self._extract_out:
            return out
        else:
            return hidden


class Reshape1DTo2D(nn.Module):
    def __init__(self, output_shape):
        super(Reshape1DTo2D, self).__init__()
        self._output_shape = output_shape

    def forward(self, x):
        x = x.view((-1, self._output_shape[0], self._output_shape[1]))
        return x


class Encoder(nn.Module):
    """
    VAE Encoder takes the input and maps it to latent representation Z
    """

    def __init__(self, z_dim, p_embedding, v_embedding, i_embedding, pr_embedding, input_shape):
        super(Encoder, self).__init__()

        self._z_dim = z_dim
        if input_shape[0] < 100:
            scale = 1
        else:
            scale = 1

        seq_len = input_shape[0]
        seq_width = input_shape[1]
        input_dim = (seq_len * seq_width)

        self._net = nn.Sequential(
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=16, batch_first=True, bidirectional=True),
            ExtractLSTMOutput(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(input_dim // 2, input_dim // 8),
        )

        self._fc_mean = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )
        self._fc_log_var = nn.Sequential(
            nn.Linear(input_dim // 8, z_dim),
        )

        # v = u.unsqueeze(-1).expand(128, 800, 4, 2048)
        # (v.reshape((2048, 800, 4, 128)) - output.unsqueeze(-1))
        # torch.norm((v.reshape((2048, 800, 4, 128)) - output.unsqueeze(-1)), dim=2)
        self._device = get_device()
        self._seq_len = seq_len
        self._seq_width = seq_width

        self._pr_embedding = pr_embedding
        self._i_embedding = i_embedding
        self._v_embedding = v_embedding
        self._p_embedding = p_embedding

    def forward(self, x):
        x_input = x.reshape((-1, self._seq_width, self._seq_len))
        h = self._net(x_input)
        mean = self._fc_mean(h)
        log_var = self._fc_log_var(h)
        return mean, log_var


class DecoderCategorical(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, output_shape, embedding, clam_output=False):
        super(DecoderCategorical, self).__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape
        seq_len = output_shape[0]
        seq_width = output_shape[1]

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 1

        output_dim = (seq_len * seq_width) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim * scale),
            nn.Dropout(0.4),
            Reshape1DTo2D((seq_width, seq_len)),
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=4, batch_first=True),
            ExtractLSTMOutput()
        )
        self._seq_len = seq_len
        self._seq_width = seq_width
        self._embedding = embedding
        self._clamp_output = clam_output

    @staticmethod
    def get_classification_from_output(output, embedding):
        reshape = False
        if output.shape[0] == MAX_MIDI_ENCODING_ROWS:
            output = output.reshape(-1, MAX_MIDI_ENCODING_ROWS)
            output = output.unsqueeze(-1)
            reshape = True
        w = embedding.weight.data
        # Expand to output size, without memory increase
        w = w.expand(-1, output.shape[1])
        w = w.unsqueeze(-1)
        # dist = w.expand(128, output.shape[1], output.shape[0]).reshape(
        #    (output.shape[0], output.shape[1], -1)) - output
        dist = torch.abs((w - output[:, None, :]))
        classification = torch.argmin(dist, dim=1)

        if reshape:
            classification = classification.reshape(MAX_MIDI_ENCODING_ROWS, -1)
        return classification.type(torch.float32)

    def forward(self, z):
        output = self._net(z)
        if self._clamp_output:
            # Clamp output in (0, 1) to prevent errors in BCE
            output = torch.clamp(output, 1e-8, 1 - 1e-8)
        else:
            output = output.reshape((-1, self._seq_len, self._seq_width))

        classification = DecoderCategorical.get_classification_from_output(output, self._embedding)
        return classification, output

    @property
    def z_dim(self):
        return self._z_dim


class Decoder(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, output_shape, clam_output=False):
        super(Decoder, self).__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape
        seq_len = output_shape[0]
        seq_width = output_shape[1]

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 1

        output_dim = (seq_len * seq_width) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim * scale),
            nn.Dropout(0.2),
            Reshape1DTo2D((seq_width, seq_len)),
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=4, batch_first=True),
            ExtractLSTMOutput(),
            nn.Tanh()
        )
        self._seq_len = seq_len
        self._seq_width = seq_width
        self._clamp_output = clam_output

    def forward(self, z):
        output = self._net(z)
        if self._clamp_output:
            # Clamp output in (0, 1) to prevent errors in BCE
            output = torch.clamp(output, 1e-8, 1 - 1e-8)
        else:
            output = output.reshape((-1, self._seq_len, self._seq_width))
        return output

    @property
    def z_dim(self):
        return self._z_dim


class ControllableDecoder(nn.Module):
    """
    VAE Decoder takes the latent Z and maps it to output of shape of the input
    """

    def __init__(self, z_dim, partial_shape, output_shape, clam_output=False):
        super(ControllableDecoder, self).__init__()
        self._z_dim = z_dim
        self._output_shape = output_shape
        seq_len = output_shape[0] - partial_shape[0]
        seq_width = output_shape[1]

        if output_shape[0] < 100:
            scale = 1
        else:
            scale = 1

        output_dim = (seq_len * seq_width) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim + partial_shape[0] * partial_shape[1], output_dim * scale),
            nn.Dropout(0.2),
            Reshape1DTo2D((seq_width, seq_len)),
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=16, batch_first=True),
            ExtractLSTMOutput(),
            nn.Tanh()
        )
        self._seq_len = seq_len
        self._seq_width = seq_width
        self._clamp_output = clam_output

        self._partial_len = partial_shape[0]
        self._partial_width = partial_shape[1]

    def forward(self, z, x_partial):
        x_flat = torch.flatten(x_partial, start_dim=1)
        input = torch.cat((z, x_flat), dim=1)
        output = self._net(input)
        if self._clamp_output:
            # Clamp output in (0, 1) to prevent errors in BCE
            output = torch.clamp(output, 1e-8, 1 - 1e-8)
        else:
            output = output.reshape((-1, self._seq_len, self._seq_width))

        output = torch.cat((x_partial, output), dim=1)
        return output

    @property
    def z_dim(self):
        return self._z_dim


class SimpleVae(BaseModel):
    def __init__(self, z_dim=64, input_shape=(10000, 8), alpha=1.0, *args: Any, **kwargs: Any):
        super(SimpleVae, self).__init__(*args, **kwargs)

        self._encoder = Encoder(z_dim,
                                input_shape=input_shape)

        self._velocity_decoder = DecoderCategorical(z_dim, self._velocities_shape, self._velocity_embedding)
        self._start_times_decoder = Decoder(z_dim, self._start_times_shape)
        self._end_times_decoder = Decoder(z_dim, self._end_times_shape)
        self._instruments_decoder = DecoderCategorical(z_dim, self._instruments_shape, self._instrument_embedding)
        self._program_decoder = DecoderCategorical(z_dim, self._programs_shape, self._program_embedding)

        self._controllable_decoder = ControllableDecoder(z_dim, (input_shape[0] // 3, input_shape[1]), input_shape)
        self._alpha = alpha
        self._model_prefix = "SimpleVaeMidi"
        self._z_dim = z_dim
        self._device = get_device()

    @staticmethod
    def from_pretrained(checkpoint_path, z_dim, input_shape):
        print(f"Loading from {checkpoint_path}...")
        _model = SimpleVae(z_dim=z_dim, input_shape=input_shape)
        _model.load_state_dict(torch.load(checkpoint_path))
        return _model

    def forward(self, x):
        mean, log_var = self._encoder(x)
        z = SimpleVae.reparameterize(mean, log_var)

        x_hat = torch.vstack(
            (x_p.T, x_v.T, x_i.T, x_programs.T, x_start_times.T, x_end_times.T)).T

        x_control = self._controllable_decoder(z, x[:, 0: x.shape[1] // 3])
        return z, x_hat, x_hat_bp, x_control, mean, log_var

    def loss_function(self, x_hat_bp, x_hat, x, x_control, mu, q_log_var):
        recon_loss = func.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
        kl = self._kl_simple(mu, q_log_var)
        loss = recon_loss + self.alpha * kl
        return loss, kl, recon_loss

    def step(self, batch, batch_idx):
        x = batch
        z, x_hat, x_hat_bp, x_control, mu, q_log_var = self(x)
        loss = self.loss_function(x_hat_bp, x_hat, x, x_control, mu, q_log_var)
        return loss

    @staticmethod
    def plot_image_grid(samples):
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(4, 4),
                         axes_pad=0.1,
                         )

        for ax, im in zip(grid, samples):
            ax.imshow(im)

        plt.show()

    def compute_fid(self):
        generated_samples = []
        n_samples = len(self._dms.val_dataloader().dataset)
        device = self._device
        print("Generating samples...")

        for i in tqdm.tqdm(range(0, n_samples)):
            rand_z = torch.randn(self._pitches_decoder.z_dim)
            rand_z_np = rand_z.numpy()
            rand_z = rand_z.to(device)

            x_pitches, _ = self._pitches_decoder(rand_z)
            x_velocity, _ = self._velocity_decoder(rand_z)
            x_start_times = self._start_times_decoder(rand_z)
            x_end_times = self._end_times_decoder(rand_z)
            x_instruments, _ = self._instruments_decoder(rand_z)
            x_programs, _ = self._program_decoder(rand_z)
            output = torch.vstack(
                (x_pitches.T, x_velocity.T, x_instruments.T, x_programs.T, x_start_times.T, x_end_times.T)).T
            generated_samples.append(output)

            # sample = output.to("cpu").detach().numpy()
            # output_dir = os.path.join(self._output_dir, f"sample-{wandb.run.name}")
            # os.makedirs(output_dir, exist_ok=True)
            # sample_file_name = os.path.join(output_dir,
            #                                f"{self._model_prefix}-{wandb.run.name}-sample-{i}.midi")
            # z_file = os.path.join(output_dir, f"z-{self._model_prefix}-{wandb.run.name}-sample-{i}.npy")
            # import numpy as np
            # np.save(z_file, rand_z_np)
            # save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)

        # Generate some controllable music
        generated_control_samples = []
        print("Generate control samples...")
        for batch_idx, batch in tqdm.tqdm(enumerate(self._dms.test_dataloader())):
            # TODO: Clean up this mess
            rand_z = torch.randn(self._pitches_decoder.z_dim)
            rand_z_np = rand_z.numpy()
            rand_z = rand_z.to(device)
            batch = batch.to(device)

            for test_sample in batch:
                control = test_sample.T.unsqueeze(0)
                control = control[:, 0:100, :]
                control_output = self._controllable_decoder(rand_z.unsqueeze(0), control)
                x = control_output

                x_pitches = DecoderCategorical.get_classification_from_output(x.T[0], self._pitch_embedding)
                x_velocity = DecoderCategorical.get_classification_from_output(x.T[1], self._velocity_embedding)
                x_instruments = DecoderCategorical.get_classification_from_output(x.T[2], self._instrument_embedding)
                x_programs = DecoderCategorical.get_classification_from_output(x.T[3], self._program_embedding)
                output = torch.stack(
                    (x_pitches, x_velocity, x_instruments, x_programs, control_output.T[4], control_output.T[5])).T
                generated_control_samples.append(output)
            # sample = output.to("cpu").detach().numpy()
            # sample = sample.reshape((300, 6))

            # output_dir = os.path.join(self._output_dir, f"sample-control-{wandb.run.name}")
            # os.makedirs(output_dir, exist_ok=True)

            # z_file = os.path.join(output_dir, f"z-{self._model_prefix}-{wandb.run.name}-sample-{batch_idx}.npy")
            # import numpy as np
            # np.save(z_file, rand_z_np)

            # sample_file_name = os.path.join(output_dir,
            # f"{self._model_prefix}-control-{wandb.run.name}-{batch_idx}.midi")
            # save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)
            if batch_idx > n_samples:
                break

        validation_samples = []
        for batch_idx, batch in tqdm.tqdm(enumerate(self._dms.val_dataloader())):
            for x in batch:
                x[0] = DecoderCategorical.get_classification_from_output(x[0], self._pitch_embedding).squeeze(1)
                x[1] = DecoderCategorical.get_classification_from_output(x[1], self._velocity_embedding).squeeze(1)
                x[2] = DecoderCategorical.get_classification_from_output(x[2], self._instrument_embedding).squeeze(1)
                x[3] = DecoderCategorical.get_classification_from_output(x[3], self._program_embedding).squeeze(1)

            validation_samples.append(batch)

            if batch_idx > n_samples:
                break

        from eval.fid_evaluator import calculate_fid
        generated_fid_score = calculate_fid(validation_samples, generated_samples)
        generated_control_fid_score = calculate_fid(validation_samples, generated_control_samples)
        print(f"Generated FID scores = {generated_fid_score}")
        print(f"Control FID scores = {generated_control_fid_score}")
        wandb.log({
            'fid': generated_fid_score,
            'controlled_generation_fid': generated_control_fid_score
        }
        )

    def sample_output(self, epoch):
        try:
            with torch.no_grad():
                device = get_device()
                # 16 for 4x4 set of numbers
                rand_z = torch.randn(16, self._decoder.z_dim).to(device)
                rand_z.to(device)
                output = self._decoder(rand_z)
                samples = output.to("cpu").detach().numpy()
                samples = samples.reshape((-1, 28, 28))
                SimpleVae.plot_image_grid(samples)
        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")

    @property
    def alpha(self):
        return self._alpha


if __name__ == "__main__":
    print(f"Training simple VAE")
    _batch_size = 2048
    _alpha = 15
    _z_dim = 20
    _model = SimpleVae(
        alpha=_alpha,
        z_dim=_z_dim,
        input_shape=(28, 28),
        use_mnist_dms=True,
        sample_output_step=10,
        batch_size=_batch_size
    )
    print(f"Training --> {_model}")

    _max_epochs = 100
    wandb.config = {
        "learning_rate": _model.lr,
        "z_dim": _z_dim,
        "epochs": _max_epochs,
        "batch_size": _batch_size,
        "alpha": _model.alpha
    }

    _optimizer = _model.configure_optimizers()
    _model.setup()
    for _epoch in elbo.elbo.ElboEpochIterator(range(1, _max_epochs + 1), _model):
        _model.train()
        _model.fit(_epoch, _optimizer)
        _model.eval()
        _model.test()
        if _epoch == 1:
            _model.compute_fid()

        if _epoch % 10 == 0:
            _model.eval()
            _model.compute_fid()
            _model.sample_output(_epoch)
            _model.save(_epoch)
