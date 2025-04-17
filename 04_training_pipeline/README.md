## MNIST: a model

### Pytorch models

First of all, pytorch models inherits from `torch.Module` class, which has the next methods:

- __init__(...) # Most of the time here we have as input: input_size, hidden_size, dropout, num classes etc
-- Also init method defines all the module architecture
- forward(self, x) # This method executes the forward pass method

Here we have an example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, num_classes) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation after first layer
        x = self.fc2(x)          # Output layer (logits)
        return x

# Instantiate the model
input_size = 20      # for example, 20 input features
hidden_size = 64     # size of the hidden layer
num_classes = 3      # number of output classes

model = SimpleClassifier(input_size, hidden_size, num_classes)

# Print model architecture
print(model)
 
```