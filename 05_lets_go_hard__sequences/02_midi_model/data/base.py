from typing import List, Dict

from pandas import DataFrame

from shared.midi.files import from_folder

class Base:
    """
    Extract all notes from each instrument of a midi file and convert them into pandas DataFrame.
    Features: [midi_idx, instrument_idx, step, pitch, velocity, duration]
    """

    """
    Extract all notes and convert them into tensors
    
    Also extracts the max and min step and duration
    """
    def __init__(self, midi_folder: str):
        self.midi_folder = midi_folder

        self.data = self.load_data()

    def load_data(self) -> DataFrame:
        """
        Load midi files from the folder and extract notes with steps.
        """
        midis = from_folder(self.midi_folder)

        data = []

        for (midi_idx, midi) in enumerate(midis):
            for (instrument_idx, instrument) in enumerate(midi.instruments):
                for (i, note) in enumerate(instrument.notes):
                    step = note.start - (instrument.notes[i - 1].start if i > 0 else 0)
                    if step < 0:
                        step = 0.1  # Avoid negative steps
                    pitch = note.pitch
                    velocity = note.velocity
                    duration = note.end - note.start

                    data.append({
                        "instrument_idx": instrument_idx,
                        "midi_idx": midi_idx,
                        "step": step,
                        "pitch": pitch,
                        "velocity": velocity,
                        "duration": duration
                    })

        return DataFrame(data)