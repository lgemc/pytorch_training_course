import unittest
import os
import struct

from libs.images import print_image_as_char
# from libs.images import convert_and_show_image
from libs.int import ubyte_to_uint

main_folder = "stubs"

class TestMNIST(unittest.TestCase):
    def test_load_images_ubyte(self):
        with open(os.path.join(main_folder, "train-images.idx3-ubyte"), 'rb') as file:
            magic, num_images, image_width, image_height = struct.unpack(">IIII", file.read(16))
            print(f"Magic: {magic}, Num Images: {num_images}, Rows: {image_height}, Cols: {image_width}")
            first_image = file.read(image_width * image_height)

            print_image_as_char(first_image, width=image_width, height=image_height)
            # convert_and_show_image(first_image, width=image_width, height=image_height)
            print()
            second_image = file.read(image_width * image_height)

            print_image_as_char(second_image, width=image_width, height=image_height)
            # convert_and_show_image(second_image, width=image_width, height=image_height)

    def test_load_labels(self):
        with open(os.path.join(main_folder, "train-labels.idx1-ubyte"), 'rb') as file:
            magic, num_images = struct.unpack(">II", file.read(8))
            print(f"Magic: {magic}, Num labels: {num_images}")

            first_label = file.read(1)
            print(f"First label: {first_label}")
            print(f"First label (int): {ubyte_to_uint(first_label)}")
            self.assertEqual(5, ubyte_to_uint(first_label), "First label should be 5")

    def test_size_of_uint(self):
        self.assertEqual(4,  struct.calcsize("I"), "Size of uint should be 4 bytes")