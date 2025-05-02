import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

def train(
        model,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epochs=50,
        learning_rate = 0.0001,
        batch_size = 64,
        device="cpu"):
    model.to(device)

    total_loss = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = {k: v.to(device) for k,v in y.items()}


            optimizer.zero_grad()

            out = model(x)
            loss_pitch = F.cross_entropy(out["pitch"], y["pitch"])
            loss_velocity = F.cross_entropy(out["velocity"], y["velocity"])
            loss_step = F.mse_loss(out["step"].squeeze(-1), y["step"])
            loss_duration = F.mse_loss(out["duration"].squeeze(-1), y["duration"])

            loss = loss_step + loss_velocity + loss_pitch + loss_duration

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        total_loss = 0
        evaluate(model, test_loader, device)


def evaluate(
        model,
        test_dataloader: DataLoader,
        device="cpu"
):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = {k: v.to(device) for k, v in x.items()}
            y = {k: v.to(device) for k, v in y.items()}

            out = model(x)

            loss_pitch = F.cross_entropy(out["pitch"], y["pitch"])
            loss_velocity = F.cross_entropy(out["velocity"], y["velocity"])
            loss_step = F.mse_loss(out["step"].squeeze(-1), y["step"])
            loss_duration = F.mse_loss(out["duration"].squeeze(-1), y["duration"])

            loss = loss_step + loss_velocity + loss_pitch + loss_duration
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss