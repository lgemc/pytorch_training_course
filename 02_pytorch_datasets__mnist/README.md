## MNIST: a data loader in pytorch

At the previous lesson we lean about mnist dataset from uint format and how to read it using python struct module.

Now we are going to practice how to convert this dataset into pytorch tensors.

## Pytorch dataset base:

Torch provides a base dataset `torch.utils.data.Dataset` which contains three main methods:
- `__len__(self)`: returns the number of samples in the dataset
- `__getitem__(self, idx)`: returns a sample from the dataset