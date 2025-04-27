import unittest
import os

import numpy as np
import pretty_midi

from note_with_step import get_notes_with_steps

main_path = "../stubs/chopin"

class TestNoteWithStep(unittest.TestCase):
    def test_note_with_step(self):
        song = pretty_midi.PrettyMIDI(os.path.join(main_path, "chp_op18.mid"))

        right_piano = song.instruments[0]

        notes = right_piano.notes

        first_note = notes[0]
        self.assertEqual(70, first_note.pitch)
        self.assertEqual(0, first_note.start)
        self.assertEqual(np.float64(0.5767279999999999), first_note.end)
        self.assertEqual(80, first_note.velocity)

        notes_with_steps = get_notes_with_steps(notes)
        first_note_with_step = notes_with_steps[0]

        self.assertEqual(first_note.pitch, first_note_with_step.pitch)
        self.assertEqual(first_note.start, first_note_with_step.start)
        self.assertEqual(first_note.end, first_note_with_step.end)
        self.assertEqual(first_note.velocity, first_note_with_step.velocity)
        self.assertEqual(first_note_with_step.step, first_note.start)

        second_note_with_step = notes_with_steps[1]
        self.assertEqual(second_note_with_step.step, second_note_with_step.start - first_note.start)
        self.assertEqual(second_note_with_step.original_note, notes[1])
        self.assertEqual(second_note_with_step.start, notes[1].start)
        self.assertEqual(second_note_with_step.end, notes[1].end)
        self.assertEqual(second_note_with_step.pitch, notes[1].pitch)
        self.assertEqual(second_note_with_step.velocity, notes[1].velocity)
