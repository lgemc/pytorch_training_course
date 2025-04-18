from torch.nn import Module
import torch
from torch.nn.functional import dropout


class MIDIModel(Module):
    def __init__(self,
                 hidden_size=128,
                 pitch_vocab_size=128,
                 velocity_vocab_size=128,
                 step_input_dim=1,
                 dropout=0.5,
                 normalize_steps=True
                 ):
        super(MIDIModel, self).__init__()
        self.velocity_embed = torch.nn.Embedding(128, velocity_vocab_size)
        self.pitch_embed = torch.nn.Embedding(128, pitch_vocab_size)
        self.normalize_steps = normalize_steps
        self.step_max = 0
        self.step_min = 0
        self.duration_max = 0
        self.duration_min = 0
        self.lstm = torch.nn.LSTM(
            input_size=step_input_dim + pitch_vocab_size + velocity_vocab_size + 1, # +1 for duration
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.pitch_out = torch.nn.Linear(hidden_size, pitch_vocab_size)
        self.velocity_out = torch.nn.Linear(hidden_size, velocity_vocab_size)
        self.step_out = torch.nn.Linear(hidden_size, step_input_dim)
        self.duration_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        pitch_embed = self.pitch_embed(x["pitch"])
        velocity_embed = self.velocity_embed(x["velocity"])
        step = x["step"].unsqueeze(-1)
        duration = x["duration"].unsqueeze(-1)
        if self.normalize_steps:
            step = (step - self.step_min) / (self.step_max - self.step_min)
            duration = (duration - self.duration_min) / (self.duration_max - self.duration_min)

        x = torch.cat((step, duration, pitch_embed, velocity_embed), dim=-1)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        x = dropout(x, p=0.5, training=self.training)

        pitch_out = self.pitch_out(x)
        velocity_out = self.velocity_out(x)
        step_out = self.step_out(x)
        duration_out = self.duration_out(x)

        return {
            "pitch": pitch_out,
            "velocity": velocity_out,
            "step": step_out,
            "duration": duration_out,
        }


    def fit_normalization(self, train_steps, train_duration):
        """Calculate normalization parameters (min, max) for step values."""
        self.step_min = train_steps.min()
        self.step_max = train_steps.max()
        self.duration_max = train_duration.max()
        self.duration_min = train_duration.min()
        if self.step_max == self.step_min:
            self.step_max = self.step_min + 1e-6

        if self.duration_max == self.duration_min:
            self.duration_max = self.duration_min + 1e-6