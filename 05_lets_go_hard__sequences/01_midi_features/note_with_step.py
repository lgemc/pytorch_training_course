from typing import List

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