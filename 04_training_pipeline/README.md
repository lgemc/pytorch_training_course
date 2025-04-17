from torch.utils.data import DataLoader

## Torch training

Torch is a library rather than a framework, so its idiomatic way to define things are functions, training pipeline 
is not the exception.

So, evaluate is a function that receives a model and perform the forward-backward process based on input data

```python
import torch

def evaluate(
        model: torch.Module, train_dataset: torch.utils.data.Dataset, test_dataset=torch.utils.data.Dataset,
        epochs: int=50, learning_rate=0.0001, device="cpu",
        batch_size=32,
):
    pass
```

## Data loader

In order to perform a training process over a hole dataset is recommended to split it into batches 
in order to have a more stable training:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

## Optimizer and loss function

Depending on your task you should select the proper optimizer and loss function, for example for classification tasks
one of the best loss functions is `torch.nn.CrossEntropyLoss()`, and a stable optimizer is adam `torch.optim.Adam(model.parameters(), lr=learning_rate)`

## The evaluation method

Here you should perform a forward pass over the validation dataset and compare predictions with real values
in order to calculate the metric that you want, for example accuracy:

```python
import torch 

def evaluate(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device="cpu",
):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
```

## The train loop
Then you should iterate per each epoch, training per batch calculating each epoch the test and validation loss

```python
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss/len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
```