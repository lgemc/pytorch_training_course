from data.base import Base
from shared.normalization import normalize
from shared.midi.note_with_step import NoteWithStep

class Normalized:
    """
    Normalized class to represent a list of notes with step and duration normalized

    It contains the features: [midi_idx, instrument_idx, step, pitch, velocity, duration]
    """
    def __init__(self, base: Base):
        self.data = base.data.copy()
        self.base = base
        self.min_duration = None
        self.max_duration = None
        self.min_step = None
        self.max_step = None

        self.normalize()

    def normalize(self):
        """
        Normalize the steps and durations.
        """
        self.min_step, self.max_step, self.data["step"] = normalize(
            self.data["step"].values
        )

        self.min_duration, self.max_duration, self.data["duration"] = normalize(
            self.data["duration"].values
        )


