import gc
import os.path
from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import utils.midi_utils
from data.midi_data_module import MAX_MIDI_ENCODING_ROWS, MIDI_ENCODING_WIDTH
from models.base.base_model import BaseModel
from utils.cuda_utils import get_device
from utils.midi_utils import save_decoder_output_as_midi
import matplotlib.pyplot as plt
import tqdm
import numpy
import wandb

wandb.init(project="music-controllable-diffusion-with-fid-version-4", entity="saravanr")


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

        self._pitch_embedding = nn.Embedding(128, 1)
        self._velocity_embedding = nn.Embedding(128, 1)
        self._instrument_embedding = nn.Embedding(128, 1)
        self._program_embedding = nn.Embedding(128, 1)

        self._encoder = Encoder(z_dim,
                                self._pitch_embedding,
                                self._velocity_embedding,
                                self._instrument_embedding,
                                self._program_embedding,
                                input_shape=input_shape)

        self._pitches_shape = (input_shape[0], 1)
        self._velocities_shape = (input_shape[0], 1)
        self._start_times_shape = (input_shape[0], 1)
        self._end_times_shape = (input_shape[0], 1)
        self._instruments_shape = (input_shape[0], 1)
        self._programs_shape = (input_shape[0], 1)

        self._pitches_decoder = DecoderCategorical(z_dim, self._pitches_shape, self._pitch_embedding)
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
        # Renorm X to midi

        mean, log_var = self._encoder(x)
        z = SimpleVae.reparameterize(mean, log_var)

        x_pitches, x_p = self._pitches_decoder(z)
        x_velocity, x_v = self._velocity_decoder(z)
        x_start_times = self._start_times_decoder(z)
        x_end_times = self._end_times_decoder(z)
        x_instruments, x_i = self._instruments_decoder(z)
        x_programs, x_pr = self._program_decoder(z)

        x_hat_bp = torch.vstack(
            (x_pitches.T, x_velocity.T, x_instruments.T, x_pr.T, x_start_times.T, x_end_times.T)).T

        x_hat = torch.vstack(
            (x_p.T, x_v.T, x_i.T, x_programs.T, x_start_times.T, x_end_times.T)).T

        x_control = self._controllable_decoder(z, x[:, 0: x.shape[1] // 3])
        return z, x_hat, x_hat_bp, x_control, mean, log_var

    def compute_midi_recon_loss(self, x, x_hat_bp, x_hat, x_control):
        # Compute losses based on variable types.
        x_pitches_hat = x_hat.T[0]
        x_velocity_hat = x_hat.T[1]
        x_instruments_hat = x_hat.T[2]
        x_programs_hat = x_hat.T[3]
        x_start_times_hat = x_hat.T[4]
        x_end_times_hat = x_hat.T[5]

        x_pitches = DecoderCategorical.get_classification_from_output(x.T[0], self._pitch_embedding)
        x_velocity = DecoderCategorical.get_classification_from_output(x.T[1], self._velocity_embedding)
        x_instruments = DecoderCategorical.get_classification_from_output(x.T[2], self._instrument_embedding)
        x_programs = DecoderCategorical.get_classification_from_output(x.T[3], self._program_embedding)

        x_pitches = x.T[0]
        x_velocity = x.T[1]
        x_instruments = x.T[2]
        x_programs = x.T[3]

        x_start_times = x.T[4]
        x_end_times = x.T[5]

        pitches_loss = func.mse_loss(x_pitches_hat, x_pitches, reduction='mean')
        velocity_loss = func.mse_loss(x_velocity_hat, x_velocity, reduction='mean')
        instruments_loss = func.mse_loss(x_instruments_hat, x_instruments, reduction='mean')
        program_loss = func.mse_loss(x_programs_hat, x_programs, reduction='mean')
        start_times_loss = func.mse_loss(x_start_times_hat, x_start_times, reduction='sum')
        end_times_loss = func.mse_loss(x_end_times_hat, x_end_times, reduction='sum')
        control_loss = func.mse_loss(x_control, x, reduction='sum')

        # TODO: How to ensure end times are > start times
        recon_loss = pitches_loss + velocity_loss + instruments_loss + program_loss + start_times_loss + end_times_loss + control_loss
        return recon_loss, pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss

    def loss_function(self, x_hat_bp, x_hat, x, x_control, mu, q_log_var):
        if self._use_mnist_dms:
            recon_loss = func.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
            pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss = [
                                                                                                                              None] * 7
        else:
            recon_loss, pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss = self.compute_midi_recon_loss(
                x, x_hat_bp, x_hat, x_control)
        kl = self._kl_simple(mu, q_log_var)
        loss = recon_loss + self.alpha * kl
        return loss, kl, recon_loss, pitches_loss, velocity_loss, instruments_loss, program_loss, start_times_loss, end_times_loss, control_loss

    def step(self, batch, batch_idx):
        x = batch
        pitch, velocity, instrument, program, start_time, duration = torch.split(x, 1, dim=1)
        s = torch.unsqueeze(start_time, dim=3)
        d = torch.unsqueeze(duration, dim=3)

        # Generate embeddings for categorical variables
        p = self._pitch_embedding(pitch.type(torch.int64))
        v = self._velocity_embedding(velocity.type(torch.int64))
        i = self._instrument_embedding(instrument.type(torch.int64))
        pr = self._program_embedding(program.type(torch.int64))

        x_input = torch.cat((p, v, i, pr, s, d), dim=3)
        x_input = torch.squeeze(x_input, dim=1)
        z, x_hat, x_hat_bp, x_control, mu, q_log_var = self(x_input)
        loss = self.loss_function(x_hat_bp, x_hat, x_input, x_control, mu, q_log_var)
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

            #sample = output.to("cpu").detach().numpy()
            #output_dir = os.path.join(self._output_dir, f"sample-{wandb.run.name}")
            #os.makedirs(output_dir, exist_ok=True)
            #sample_file_name = os.path.join(output_dir,
            #                                f"{self._model_prefix}-{wandb.run.name}-sample-{i}.midi")
            #z_file = os.path.join(output_dir, f"z-{self._model_prefix}-{wandb.run.name}-sample-{i}.npy")
            #import numpy as np
            #np.save(z_file, rand_z_np)
            #save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)

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
            #sample = output.to("cpu").detach().numpy()
            #sample = sample.reshape((300, 6))

            #output_dir = os.path.join(self._output_dir, f"sample-control-{wandb.run.name}")
            #os.makedirs(output_dir, exist_ok=True)

            #z_file = os.path.join(output_dir, f"z-{self._model_prefix}-{wandb.run.name}-sample-{batch_idx}.npy")
            #import numpy as np
            #np.save(z_file, rand_z_np)

           # sample_file_name = os.path.join(output_dir,
                                            #f"{self._model_prefix}-control-{wandb.run.name}-{batch_idx}.midi")
            #save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)
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

        #wandb.run.summary['fid'] = generated_fid_score
        #wandb.run.summary['control_fid'] = generated_control_fid_score

    def sample_output(self, epoch):
        try:
            with torch.no_grad():
                device = get_device()
                if self._use_mnist_dms:
                    # 16 for 4x4 set of numbers
                    rand_z = torch.randn(16, self._decoder.z_dim).to(device)
                    rand_z.to(device)
                    output = self._decoder(rand_z)
                    samples = output.to("cpu").detach().numpy()
                    samples = samples.reshape((-1, 28, 28))
                    SimpleVae.plot_image_grid(samples)
                else:
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

                    sample = output.to("cpu").detach().numpy()
                    sample_file_name = os.path.join(self._output_dir,
                                                    f"{self._model_prefix}-{wandb.run.name}-{epoch}.midi")
                    save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)
                    print(f"Generating midi sample file://{sample_file_name}")
                    print(f"Generating midi sample path = {sample_file_name}")

                    output_dir = os.path.join(self._output_dir, f"sample-control-{wandb.run.name}")
                    os.makedirs(output_dir, exist_ok=True)

                    z_file = os.path.join(output_dir, f"z-{self._model_prefix}-{wandb.run.name}-sample-{epoch}.npy")
                    import numpy as np
                    np.save(z_file, rand_z_np)

                    sample_file_name = os.path.join(output_dir, f"{self._model_prefix}-control-{wandb.run.name}-{epoch}.midi")
                    save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)

                    # Generate some controllable music
                    for batch_idx, batch in enumerate(self._dms.val_dataloader()):
                        # TODO: Clean up this mess
                        batch = batch.to(device)
                        control = batch[0].T.unsqueeze(0)
                        control = control[:, 0:100, :]
                        control_output = self._controllable_decoder(rand_z.unsqueeze(0), control)
                        x = control_output

                        x_pitches = DecoderCategorical.get_classification_from_output(x.T[0], self._pitch_embedding)
                        x_velocity = DecoderCategorical.get_classification_from_output(x.T[1], self._velocity_embedding)
                        x_instruments = DecoderCategorical.get_classification_from_output(x.T[2],
                                                                                          self._instrument_embedding)
                        x_programs = DecoderCategorical.get_classification_from_output(x.T[3], self._program_embedding)
                        output = torch.stack((x_pitches, x_velocity, x_instruments, x_programs, control_output.T[4],
                                              control_output.T[5])).T
                        sample = output.to("cpu").detach().numpy()
                        sample = sample.reshape((300, 6))
                        sample_file_name = os.path.join(output_dir,
                                                        f"{self._model_prefix}-control-{wandb.run.name}-{epoch}.midi")
                        save_decoder_output_as_midi(sample, sample_file_name, self._data_mean, self._data_std)
                        print(f"Generating controlled midi sample file://{sample_file_name}")
                        print(f"Generating controlled midi sample path = {sample_file_name}")
                        break

        except Exception as _e:
            print(f"Hit exception during sample_output - {_e}")

    @property
    def alpha(self):
        return self._alpha


if __name__ == "__main__":
    print(f"Training simple VAE")
    batch_size = 2048
    train_mnist = False

    _alpha = 15

    if train_mnist:
        _z_dim = 20
        model = SimpleVae(
            alpha=_alpha,
            z_dim=_z_dim,
            input_shape=(28, 28),
            use_mnist_dms=True,
            sample_output_step=10,
            batch_size=batch_size
        )
    else:
        _z_dim = 32
        model = SimpleVae(
            alpha=_alpha,
            z_dim=_z_dim,
            input_shape=(MAX_MIDI_ENCODING_ROWS, MIDI_ENCODING_WIDTH),
            use_mnist_dms=False,
            sample_output_step=10,
            batch_size=batch_size
        )
    print(f"Training --> {model}")

    max_epochs = 100
    wandb.config = {
        "learning_rate": model.lr,
        "z_dim": _z_dim,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "alpha": model.alpha
    }

    _optimizer = model.configure_optimizers()
    model.setup()
    for _epoch in range(1, max_epochs + 1):
        model.train()
        model.fit(_epoch, _optimizer)
        model.eval()
        model.test()
        if _epoch == 1:
            model.compute_fid()

        if _epoch % 10 == 0:
            model.eval()
            model.compute_fid()
            model.sample_output(_epoch)
            model.save(_epoch)

