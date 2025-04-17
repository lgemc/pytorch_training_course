import unittest
from dataset import MNIST
from libs.images import print_image_as_char
root_folder = "stubs"

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load once for all test methods
        dataset = MNIST(root_folder, split="train")
        cls.dataset = dataset

    def test_train_split(self):
        self.assertEqual(60000, len(self.dataset.labels))
        self.assertEqual((60000, 28*28), self.dataset.data.shape)

    def test_len(self):
        self.assertEqual(60000, len(self.dataset))

    def test_get_item(self):
        x, y = self.dataset[10]
        self.assertEqual(3.0, y)
        print(f"Tenth element from the dataset: {y}")
        print_image_as_char(x, 28, 28)

    def test_show_image(self):
        first_image = self.dataset.data[0]
        print_image_as_char(first_image, 28, 28)
        print(f"First label: {self.dataset.labels[0]}")
        self.assertEqual(5.0, self.dataset.labels[0])

    def test__test_split(self):
        dataset = MNIST(root_folder, split="test")
        self.assertEqual(10000, dataset.labels.shape[0])
