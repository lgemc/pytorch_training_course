import unittest
import os

import numpy as np
import pretty_midi

main_path = "../stubs/chopin"

class TestMIDI(unittest.TestCase):
    def test_read_midi(self):
        song = pretty_midi.PrettyMIDI(os.path.join(main_path, "chp_op18.mid"))

        self.assertEqual(2, len(song.instruments))
        self.assertEqual("Piano right", song.instruments[0].name)
        self.assertEqual("Piano left", song.instruments[1].name)

        right_piano = song.instruments[0]

        notes = right_piano.notes
        self.assertEqual(2098, len(notes))

        first_note = notes[0]
        self.assertEqual(70, first_note.pitch)
        self.assertEqual(0, first_note.start)
        self.assertEqual(np.float64(0.5767279999999999), first_note.end)
        self.assertEqual(80, first_note.velocity)