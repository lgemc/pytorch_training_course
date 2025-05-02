import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.sequences import PITCH_COLUMN, VELOCITY_COLUMN, DURATION_COLUMN, STEP_COLUMN


def train(
        model,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epochs=50,
        learning_rate=0.0001,
        batch_size=64,
        device="cpu"):
    model.to(device)

    total_loss = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            # Extract individual predictions
            pitch_preds = outputs['pitch']
            velocity_preds = outputs['velocity']
            step_preds = outputs['step']
            duration_preds = outputs['duration']

            # Calculate losses for each attribute
            loss_pitch = F.cross_entropy(
                pitch_preds.reshape(-1, pitch_preds.size(-1)),
                y[:, :, PITCH_COLUMN].reshape(-1).long()
            )

            loss_velocity = F.cross_entropy(
                velocity_preds.reshape(-1, velocity_preds.size(-1)),
                y[:, :, VELOCITY_COLUMN].reshape(-1).long()
            )

            loss_step = F.mse_loss(
                step_preds.squeeze(-1),
                y[:, :, STEP_COLUMN]
            )

            loss_duration = F.mse_loss(
                duration_preds.squeeze(-1),
                y[:, :, DURATION_COLUMN]
            )

            # Combine losses
            loss = loss_step + loss_velocity + loss_pitch + loss_duration

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, test_loader, device)


def evaluate(model, test_loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)

            # Extract individual predictions
            pitch_preds = outputs['pitch']
            velocity_preds = outputs['velocity']
            step_preds = outputs['step']
            duration_preds = outputs['duration']

            # Calculate losses for each attribute
            loss_pitch = F.cross_entropy(
                pitch_preds.reshape(-1, pitch_preds.size(-1)),
                y[:, :, PITCH_COLUMN].reshape(-1).long()
            )

            loss_velocity = F.cross_entropy(
                velocity_preds.reshape(-1, velocity_preds.size(-1)),
                y[:, :, VELOCITY_COLUMN].reshape(-1).long()
            )

            loss_step = F.mse_loss(
                step_preds.squeeze(-1),
                y[:, :, STEP_COLUMN]
            )

            loss_duration = F.mse_loss(
                duration_preds.squeeze(-1),
                y[:, :, DURATION_COLUMN]
            )

            # Combine losses
            loss = loss_step + loss_velocity + loss_pitch + loss_duration

            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")