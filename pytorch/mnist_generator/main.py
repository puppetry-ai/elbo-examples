import os
import os.path
from typing import Any
from typing import Optional

import elbo.elbo
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
import tqdm
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class MNISTTensorDataSet(Dataset):
    def __init__(self, tensors):
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
        self._val_dataloader = None
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
        train_x = train_x.astype(np.float32) / 255.0
        train_x = [torch.tensor(x).to(device) for x in train_x]
        full_data_set = MNISTTensorDataSet(train_x)

        test_x = np.array([np.array(x) for x, _ in datasets.MNIST(self._data_dir, train=False, download=True)])
        test_x = test_x.astype(np.float32) / 255.0
        test_x = [torch.tensor(x).to(device) for x in test_x]
        test_data_set = MNISTTensorDataSet(test_x)

        # Split train to train and val
        val_size = len(test_data_set)
        train_size = len(full_data_set) - val_size

        train_data_set, val_data_set = torch.utils.data.random_split(full_data_set, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_data_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            val_data_set,
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers)

        test_loader = torch.utils.data.DataLoader(
            test_data_set,
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers)

        self._train_dataloader = train_loader
        self._test_dataloader = test_loader
        self._val_dataloader = val_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._val_dataloader

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


class BaseModel(elbo.elbo.ElboModel, torch.nn.Module):
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
        super(BaseModel, self).__init__()
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

    def loss_function(self, x_hat, x, qm, qv):
        raise NotImplementedError(f"Please implement loss_function()")

    def step(self, batch, batch_idx):
        raise NotImplementedError(f"Please implement step()")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def sample_output(self, epoch):
        raise NotImplementedError(f"Please implement sample_output()")

    def fit(self, epoch, optimizer):
        print(f'Training mode epoch - {epoch}')
        device = get_device()
        self.to(device)
        self.train()
        batch_train_loss = 0
        batch_kl_loss = 0.0
        batch_recon_loss = 0.0

        for batch_idx, batch in enumerate(tqdm.tqdm(self._dms.train_dataloader())):
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

        print(f'====> Train Loss = {loss} KL = {kl_loss} Recon = {recon_loss}  Epoch = {epoch}')

    def save(self):
        model_save_path = os.path.join(self._output_dir,
                                       f"{self._model_prefix}.checkpoint")
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

    def __init__(self, z_dim, input_shape):
        super(Encoder, self).__init__()
        self._z_dim = z_dim
        seq_len = input_shape[0]
        seq_width = input_shape[1]
        input_dim = (seq_len * seq_width)

        self._net = nn.Sequential(
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=6, batch_first=True, bidirectional=True),
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

        self._device = get_device()
        self._seq_len = seq_len
        self._seq_width = seq_width

    def forward(self, x):
        x_input = x.reshape((-1, self._seq_width, self._seq_len))
        h = self._net(x_input)
        mean = self._fc_mean(h)
        log_var = self._fc_log_var(h)
        return mean, log_var


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
        scale = 1
        output_dim = (seq_len * seq_width) // scale
        self._net = nn.Sequential(
            nn.Linear(z_dim, output_dim * scale),
            nn.Dropout(0.2),
            Reshape1DTo2D((seq_width, seq_len)),
            nn.GRU(input_size=seq_len, hidden_size=seq_len, num_layers=4, batch_first=True),
            ExtractLSTMOutput(),
            nn.Sigmoid()
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


class SimpleVae(BaseModel):
    def __init__(self,
                 z_dim=8,
                 input_shape=(28, 28),
                 alpha=1.0,
                 *args: Any,
                 **kwargs: Any):
        super(SimpleVae, self).__init__(*args, **kwargs)
        self._encoder = Encoder(z_dim,
                                input_shape=input_shape)
        self._decoder = Decoder(z_dim, input_shape)
        self._alpha = alpha
        self._model_prefix = "VAE_MNIST"
        self._z_dim = z_dim
        self._device = get_device()

    @staticmethod
    def from_pretrained(checkpoint_path, z_dim, input_shape):
        print(f"Loading from {checkpoint_path}...")
        model = SimpleVae(z_dim=z_dim, input_shape=input_shape)
        model.load_state_dict(torch.load(checkpoint_path))
        return model

    def forward(self, x):
        mean, log_var = self._encoder(x)

        z = SimpleVae.reparameterize(mean, log_var)
        x_hat = self._decoder(z)
        return z, x_hat, mean, log_var

    def loss_function(self, x_hat, x, mu, q_log_var):
        recon_loss = func.binary_cross_entropy(x_hat, x, reduction='sum')
        kl = self._kl_simple(mu, q_log_var)
        loss = recon_loss + self.alpha * kl
        return loss, kl, recon_loss

    def step(self, batch, batch_idx):
        x = batch
        parallel_self = nn.DataParallel(self)
        z, x_hat, mu, q_log_var = parallel_self(x)
        loss = self.loss_function(x_hat, x, mu, q_log_var)
        return loss

    @staticmethod
    def plot_image_grid(samples, tag):
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(4, 4),
                         axes_pad=0.1,
                         )

        for ax, im in zip(grid, samples):
            ax.imshow(im)

        plt.savefig(f"sample_image_{tag}.png")

    def sample_output(self, epoch):
        print(f"Sampling output {epoch}")
        try:
            with torch.no_grad():
                device = get_device()
                # 16 for 4x4 set of numbers
                rand_z = torch.randn(16, self._decoder.z_dim).to(device)
                rand_z.to(device)
                output = self._decoder(rand_z)
                samples = output.to("cpu").detach().numpy()
                samples = samples.reshape((-1, 28, 28))
                SimpleVae.plot_image_grid(samples, tag=f"{epoch}")
        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")

    @property
    def alpha(self):
        return self._alpha


def train_generator():
    print(f"Training simple VAE")
    batch_size = 10
    alpha = 1
    z_dim = 20
    model = SimpleVae(
        alpha=alpha,
        z_dim=z_dim,
        input_shape=(28, 28),
        sample_output_step=1,
        batch_size=batch_size
    )

    print(f"Training --> {model} on {get_device()}")

    max_epochs = 100
    optimizer = model.configure_optimizers()
    model.setup()
    for epoch in elbo.elbo.ElboEpochIterator(range(0, max_epochs), model):
        model.train()
        model.fit(epoch, optimizer)
        model.eval()
        if epoch % 10 == 0:
            model.eval()
            model.sample_output(epoch)
            model.save()


if __name__ == "__main__":
    train_generator()
