from typing import List, Tuple
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
import numpy as np
import pretty_midi
from data.normalized import Normalized
from shared.sequences import build as build_sequence

STEP_COLUMN = 0
PITCH_COLUMN = 1
VELOCITY_COLUMN = 2
DURATION_COLUMN = 3

class Sequenced(Dataset):
    """
    Build ngrams from midi files.

    It also extracts the steps
    and normalizes them.
    It also extracts the durations
    and normalizes them.

    data
    """
    sequences = [([], [])]

    def __init__(self, data=Normalized, sequence_size=5):
        self.normalized = data
        self.sequence_size = sequence_size
        self.sequences = self._build_sequences_per_instrument(
            self._build_notes_per_instrument(
                self.normalized.data
            )
        )

    def _build_notes_per_instrument(self, data: DataFrame):
        grouped = data.groupby(["instrument_idx", "midi_idx"])

        notes_per_instrument = []
        for _, group in grouped:
            # Extract the features
            features = group[["step", "pitch", "velocity", "duration"]].values
            notes_per_instrument.append(features)

        return notes_per_instrument

    def _build_sequences_per_instrument(self, notes_per_instrument: List[List]) -> torch.Tensor:
        """
        Build sequences from the notes per instrument
        """
        sequences = []

        for notes in notes_per_instrument:
            instrument_sequences = build_sequence(notes, self.sequence_size)
            sequences.extend(instrument_sequences)

        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Get the item at the index idx
        """
        return self.sequences[idx]


def items_to_notes(
    items: torch.Tensor,
    step_max: float = None,
    step_min: float = None,
    duration_max: float = None,
    duration_min: float = None,
    prev_note: pretty_midi.Note = None,
) -> List[pretty_midi.Note]:
    """
    Convert a tensor from the dataset to a pretty_midi.Note objects
    """
    pitches = items[:, PITCH_COLUMN]
    velocities = items[:, VELOCITY_COLUMN]
    durations = items[:, DURATION_COLUMN]
    durations = durations * (duration_max - duration_min) + duration_min
    steps = items[:, STEP_COLUMN]
    steps = steps * (step_max - step_min) + step_min

    notes = []
    for i in range(len(pitches)):
        pitch = pitches[i].item()
        velocity = velocities[i].item()
        duration = durations[i].item()
        step = steps[i].item()

        start = prev_note.start + step if prev_note else 0
        end = start + duration

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )

        prev_note = note

        notes.append(note)
    return notes