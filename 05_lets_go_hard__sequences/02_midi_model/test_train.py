import unittest
import torch

from model import MIDIModel
from dataset import MIDIDataset
from train import train_test_split_sequential, train

notes_folder = "../stubs/chopin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestTrain(unittest.TestCase):
    def test_train_testdata_split(self):
        dataset = MIDIDataset(notes_folder)

        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)
        self.assertEqual(len(train_dataset) + len(test_dataset), len(dataset))
        self.assertEqual(len(train_dataset), len(dataset) * 0.8)
        self.assertEqual(len(test_dataset), len(dataset) * 0.2)

    def test_train(self):
        model = MIDIModel(pitch_vocab_size=128, dropout=0.2)
        model.to(device)
        dataset = MIDIDataset(notes_folder)
        model.fit_normalization(dataset.get_steps().to(device), dataset.get_durations().to(device))

        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)
        train(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=30,
            batch_size=20,
            learning_rate=0.0001,
            device=device,
        )

        torch.save(model.state_dict(), "model.pth")
