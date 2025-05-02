import os
from typing import List

import pretty_midi


def from_folder(folder_path) -> List[pretty_midi.PrettyMIDI]:
    midis = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid"):
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(folder_path, file))
                midis.append(midi)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return midis


def export(notes: List[pretty_midi.Note], out_file: str):
    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    instrument.notes = notes

    midi.instruments.append(instrument)
    for note in notes:
        print(f"{note.pitch}, {note.start}, {note.end}, {note.velocity}")

    midi.write(out_file)