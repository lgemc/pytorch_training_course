import torch
from torch.utils.data import Dataset

from data.base import Base
from data.normalized import Normalized
from data.sequences import Sequenced, PITCH_COLUMN, VELOCITY_COLUMN, DURATION_COLUMN, STEP_COLUMN

class MIDIDataset(Dataset):
    def __init__(self, midis_folder: str, sequence_size: int = 5):
        """
        Initialize the dataset with a folder containing MIDI files.
        """
        self.base = Base(midis_folder)
        self.normalized = Normalized(self.base)
        self.sequenced = Sequenced(self.normalized, sequence_size=sequence_size)


    def __len__(self):
        return len(self.sequenced)

    def __getitem__(self, idx: int):
        """
        Get a sequence and its corresponding target.
        """
        x, y = self.sequenced[idx]

        # Convert to tensors
        return x, y