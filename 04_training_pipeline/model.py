import torch

class Model(torch.nn.Module):
    def __init__(self, num_features=28*28, hidden_size=32, num_classes =10):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x