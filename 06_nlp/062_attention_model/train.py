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

def train(
        model,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        lr=0.0001,
        epochs=400,
        batch_size=12,
        device="cpu"):
    model.to(device)
    model.train()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0  # Reset total_loss for the current epoch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss_value = loss(out, y)
            total_loss += loss_value.item()

            loss_value.backward()
            optim.step()


        print(f"Epoch {epoch}, train loss: {total_loss/len(train_loader)}:.4f")
        # evaluate(model, test_loader, device)


