from typing import List, Tuple
from torch.utils.data import Dataset
import torch

from lib import build_ngrams, read_midi_files, get_notes_with_steps

class MIDIDataset(Dataset):
    """
    Build ngrams from midi files.

    It also extracts the steps
    and normalizes them.
    It also extracts the durations
    and normalizes them.
    """
    def __init__(self, midi_folder: str, n_gram_size=5):
        self.midi_folder = midi_folder
        self.ngram_size = n_gram_size
        self._ngrams = None
        self._midis = None
        self._steps = []
        self._durations = []
        self._notes = []
        self.step_min = None
        self.step_max = None
        self.duration_max = None
        self.duration_min = None

        self._load_ngrams()

        self._setup_normalization(
            torch.tensor(self._steps, dtype=torch.float32),
            torch.tensor(self._durations, dtype=torch.float32)
        )

    def _load_ngrams(self):
        midi_files = read_midi_files(self.midi_folder)
        self._midis = midi_files
        n_grams = self._build_ngrams(midi_files)
        self._ngrams = n_grams

    def _build_ngrams(self, midi_files) -> List[Tuple[List, any]]:
        """
        Build ngrams from the midi files.
        It also extracts the steps
        """
        all_ngrams = []
        for midi_file in midi_files:
            for instrument in midi_file.instruments:
                notes = get_notes_with_steps(instrument.notes)
                n_grams = build_ngrams(notes, self.ngram_size)
                all_ngrams.extend(n_grams)
                for note in notes:
                    self._steps.append(note.step)
                    self._notes.append(note)
                    self._durations.append(note.duration)


        return all_ngrams

    def __len__(self):
        return len(self._ngrams)

    def get_steps(self):
        return torch.tensor(self._steps, dtype=torch.float32)

    def get_durations(self):
        return torch.tensor(self._durations, dtype=torch.float32)

    def __getitem__(self, idx):
        context, target = self._ngrams[idx]

        steps = torch.tensor([note.step for note in context], dtype=torch.float32)
        durations = torch.tensor([note.duration for note in context], dtype=torch.float32)
        steps, durations = self._normalize(steps, durations)

        x = {
            "step": steps,
            "velocity": torch.tensor([note.velocity for note in context], dtype=torch.long),
            "pitch": torch.tensor([note.pitch for note in context], dtype=torch.long),
            "duration": durations,
        }

        y = {
            "step": torch.tensor(target.step, dtype=torch.float32),
            "velocity": torch.tensor(target.velocity, dtype=torch.long),
            "pitch": torch.tensor(target.pitch, dtype=torch.long),
            "duration": torch.tensor(target.duration, dtype=torch.float32)
        }

        return x, y

    def _setup_normalization(self, steps, duration):
        """Calculate normalization parameters (min, max) for step values."""
        self.step_min = steps.min()
        self.step_max = steps.max()
        self.duration_max = duration.max()
        self.duration_min = duration.min()
        if self.step_max == self.step_min:
            self.step_max = self.step_min + 1e-6

        if self.duration_max == self.duration_min:
            self.duration_max = self.duration_min + 1e-6


    def _normalize(self, steps, durations):
        """Normalize the steps and durations."""
        steps = (steps - self.step_min) / (self.step_max - self.step_min)
        durations = (durations - self.duration_min) / (self.duration_max - self.duration_min)
        return steps, durations