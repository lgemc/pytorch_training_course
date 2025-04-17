import os
import struct
import torch
from torch.utils.data import Dataset

from libs.int import ubyte_to_uint

train_images_file_name = 'train-images.idx3-ubyte'
test_images_file_name = 't10k-images.idx3-ubyte'

train_labels_file_name = 'train-labels.idx1-ubyte'
test_labels_file_name = 't10k-labels.idx1-ubyte'

class MNIST(Dataset):
    def __init__(self, folder, split="train"):
        self.folder = folder
        self.data = None # an array of arrays containing all images
        self.image_width = None
        self.image_height = None
        self._num_images = None
        self.labels = None

        images_file = train_images_file_name if split == "train" else test_images_file_name
        labels_file = train_labels_file_name if split == "train" else test_labels_file_name

        self.data = self._read_images(images_file)
        self.labels = self._read_labels(labels_file)

    def _read_images(self, file) -> torch.Tensor:
        try:
            file = open(os.path.join(self.folder, file), "rb")
        except Exception as e:
            raise e


        _, num_images, image_width, image_height = struct.unpack(">IIII", file.read(16))

        images_data = file.read(num_images * image_width * image_height)

        images = torch.tensor(list(images_data), dtype=torch.float32) / 255 # normalized data

        file.close()

        return images.view((num_images, image_width*image_height))

    def _read_labels(self, file: str) -> torch.Tensor:
        try:
            file = open(os.path.join(self.folder, file), "rb")
        except Exception as e:
            raise e


        _, num_labels = struct.unpack(">II", file.read(8))
        labels = []
        for i in range(num_labels):
            raw_label = ubyte_to_uint(file.read(1))
            labels.append(raw_label)

        file.close()

        return torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.labels[idx]

        return x, y