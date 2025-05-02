import unittest

from data.base import Base

midis_folder = "../../stubs/chopin"

class TestDataset(unittest.TestCase):
    def test__init(self):
        dataset = Base(midis_folder)