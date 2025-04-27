import torch

def evaluate(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device="cpu",
):
    model.eval()

    total_loss = 0
    num_iter = 0

    loss = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_loader:
            num_iter += 1
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss_value = loss(outputs, y)

            total_loss += loss_value.item() / len(test_loader)

    return total_loss / num_iter

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
        num_iter = 0
        for x, y in train_loader:
            num_iter += 1
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss_value = loss(out, y)
            total_loss += loss_value.item()

            loss_value.backward()
            optim.step()

        eval_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}, train loss: {total_loss / num_iter}, eval loss: {eval_loss}")



