from torch.nn import Module
import torch
from torch.nn.functional import dropout


class MIDIModel(Module):
    def __init__(self,
            hidden_size=128,
            pitch_vocab_size=128,
            velocity_vocab_size=128,
            step_input_dim=1,
            dropout_p=0.5,
            step_max=0,
            step_min=0,
            duration_max=0,
            duration_min=0,
    ):
        super(MIDIModel, self).__init__()
        self.velocity_embed = torch.nn.Embedding(128, velocity_vocab_size)
        self.pitch_embed = torch.nn.Embedding(128, pitch_vocab_size)

        self.step_max = step_max
        self.step_min = step_min
        self.duration_max = duration_max
        self.duration_min = duration_min

        self.lstm = torch.nn.LSTM(
            input_size=step_input_dim + pitch_vocab_size + velocity_vocab_size + 1, # +1 for duration
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_p,
        )

        self.pitch_out = torch.nn.Linear(hidden_size, pitch_vocab_size)
        self.velocity_out = torch.nn.Linear(hidden_size, velocity_vocab_size)
        self.step_out = torch.nn.Linear(hidden_size, step_input_dim)
        self.duration_out = torch.nn.Linear(hidden_size, 1)
        self.dropout = dropout_p

    def forward(self, x):
        pitch_embed = self.pitch_embed(x["pitch"])
        velocity_embed = self.velocity_embed(x["velocity"])
        step = x["step"].unsqueeze(-1)
        duration = x["duration"].unsqueeze(-1)

        x = torch.cat((step, duration, pitch_embed, velocity_embed), dim=-1)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        x = dropout(x, p=self.dropout, training=self.training)

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

    def state_dict(self, *args, **kwargs):
        # Get the default state_dict
        state = super().state_dict(*args, **kwargs)
        # Add step_max and step_min to the state_dict
        state["step_max"] = self.step_max
        state["step_min"] = self.step_min
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Load step_max and step_min from the state_dict
        self.step_max = state_dict.pop("step_max", 0)
        self.step_min = state_dict.pop("step_min", 0)
        # Load the remaining state_dict
        super().load_state_dict(state_dict, strict)