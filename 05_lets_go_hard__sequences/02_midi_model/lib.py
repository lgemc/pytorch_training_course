from typing import List, Tuple
import os

import pretty_midi

class NoteWithStep:
    """
    NoteWithStep class to represent a note with its step and original note.
    Step is the time difference between the current note and the previous note (note[i].start - note[i-1].start).
    """
    def __init__(self, start, end, pitch, velocity, step, original_note=None):
        self.original_note = original_note
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.step = step
        self.duration = self.end - self.start


def get_notes_with_steps(notes: List):
    all_notes = []
    """Extract notes with step information from instruments."""
    for i, note in enumerate(notes):
        if i == 0:
            all_notes.append(NoteWithStep(
                note.start, note.end, note.pitch, note.velocity, note.start, note
            ))
        else:
            step = note.start - notes[i - 1].start
            all_notes.append(NoteWithStep(
                note.start, note.end, note.pitch, note.velocity, step, note
            ))

    return all_notes

def build_ngrams(elements: List, n=5) -> List[Tuple[List, any]]:
    ngrams = []
    for i in range(len(elements) - n):
        ngram = elements[i:i + n]
        ngrams.append((ngram, elements[i + n]))

    return ngrams

def read_midi_files(folder_path):
    """Read all MIDI files in a folder and return a list of PrettyMIDI objects."""
    midis = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid"):
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(folder_path, file))
                midis.append(midi)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return midis

def write_midi(notes: List[pretty_midi.Note], out_file: str):
    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    instrument.notes = notes

    midi.instruments.append(instrument)
    for note in notes:
        print(f"{note.pitch}, {note.start}, {note.end}, {note.velocity}")
    midi.write(out_file)