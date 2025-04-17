import unittest
from int import ubyte_to_uint

class TestInt(unittest.TestCase):
    def test_convert_uint(self):
        byte = b'\x01'
        expected_uint = 1
        result = ubyte_to_uint(byte)
        self.assertEqual(result, expected_uint)

        byte = b'\x02'
        expected_uint = 2
        result = ubyte_to_uint(byte)
        self.assertEqual(result, expected_uint)

        expected_uint = 255
        byte = b'\xff'
        result = ubyte_to_uint(byte)
        self.assertEqual(result, expected_uint)