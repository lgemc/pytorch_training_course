import torch
from torch import nn
from data.sequences import PITCH_COLUMN, VELOCITY_COLUMN, DURATION_COLUMN, STEP_COLUMN


class MusicSeq2SeqModel(nn.Module):
    def __init__(
            self,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            pitch_vocab_size=128,
            velocity_vocab_size=128,
            pitch_embed_dim=128,
            velocity_embed_dim=128,
            step_input_dim=1,
            dropout_p=0.5,
            step_max=0,
            step_min=0,
            duration_max=0,
            duration_min=0,
    ):
        super(MusicSeq2SeqModel, self).__init__()
        self.velocity_embed = torch.nn.Embedding(velocity_vocab_size, velocity_embed_dim)
        self.pitch_embed = torch.nn.Embedding(pitch_vocab_size, pitch_embed_dim)
        self.pitch_norm = torch.nn.LayerNorm(pitch_embed_dim)
        self.velocity_norm = torch.nn.LayerNorm(velocity_embed_dim)

        self.input_dim = step_input_dim + pitch_embed_dim + velocity_embed_dim + 1# + 1 (for duration)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.step_max = step_max
        self.step_min = step_min
        self.duration_max = duration_max
        self.duration_min = duration_min

        # Encoder (LSTM)
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Decoder (LSTM)
        self.decoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Output layers
        self.pitch_out = nn.Linear(hidden_dim, pitch_vocab_size)
        self.velocity_out = nn.Linear(hidden_dim, velocity_vocab_size)
        self.step_out = nn.Linear(hidden_dim, step_input_dim)
        self.duration_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, teacher_forcing_ratio=0.5):
        pitch_embed = self.pitch_embed(x[:, :, PITCH_COLUMN].to(torch.long))
        velocity_embed = self.velocity_embed(x[:, :, VELOCITY_COLUMN].to(torch.long))
        pitch_embed = self.pitch_norm(pitch_embed)
        velocity_embed = self.velocity_norm(velocity_embed)
        step = x[:, :, STEP_COLUMN].unsqueeze(-1)
        duration = x[:, :, DURATION_COLUMN].unsqueeze(-1)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Concatenate all features for encoder input
        encoder_input = torch.cat((step, duration, pitch_embed, velocity_embed), dim=-1)

        # Pass through encoder
        _, (hidden, cell) = self.encoder(encoder_input)

        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=x.device)

        # Initialize output tensors
        pitch_outputs = torch.zeros(batch_size, seq_len, 128, device=x.device)
        velocity_outputs = torch.zeros(batch_size, seq_len, 128, device=x.device)
        step_outputs = torch.zeros(batch_size, seq_len, 1, device=x.device)
        duration_outputs = torch.zeros(batch_size, seq_len, 1, device=x.device)

        # Decode sequence step by step
        for t in range(seq_len):
            # Pass through decoder
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # Get predictions for each attribute
            pitch_pred = self.pitch_out(output)
            velocity_pred = self.velocity_out(output)
            step_pred = self.step_out(output)
            duration_pred = self.duration_out(output)

            # Store predictions
            pitch_outputs[:, t:t + 1, :] = pitch_pred
            velocity_outputs[:, t:t + 1, :] = velocity_pred
            step_outputs[:, t:t + 1, :] = step_pred
            duration_outputs[:, t:t + 1, :] = duration_pred

            # Create next decoder input
            # Create one-hot encoding for pitch and velocity predictions
            _, pitch_idx = torch.max(pitch_pred, dim=-1)
            _, velocity_idx = torch.max(velocity_pred, dim=-1)

            next_pitch_embed = self.pitch_embed(pitch_idx)
            next_velocity_embed = self.velocity_embed(velocity_idx)

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            if teacher_force and t < seq_len - 1:
                # Use ground truth as next input
                decoder_input = encoder_input[:, t + 1:t + 2, :]
            else:
                # Use prediction as next input
                next_input = torch.cat((
                    step_pred,
                    duration_pred,
                    next_pitch_embed,
                    next_velocity_embed
                ), dim=-1)
                decoder_input = next_input

        return {
            'pitch': pitch_outputs,
            'velocity': velocity_outputs,
            'step': step_outputs,
            'duration': duration_outputs
        }