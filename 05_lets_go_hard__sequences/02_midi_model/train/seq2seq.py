import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.sequences import PITCH_COLUMN, VELOCITY_COLUMN, DURATION_COLUMN, STEP_COLUMN

def diversity_loss(predictions):
    # Penalize low entropy distributions
    probs = F.softmax(predictions, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return -torch.mean(entropy)  # Negative because we want to maximize entropy


def train(
        model,
        train_dataset: Dataset,
        test_dataset: Dataset,
        epochs=50,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        device="cpu"):
    model.to(device)

    total_loss = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            steps += 1
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

            if steps % log_interval == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")

                print("pitch ", y[0, :, PITCH_COLUMN])
                print("velocity ", y[0, :, VELOCITY_COLUMN])
                print("step ", y[0, :, STEP_COLUMN])
                print("duration ", y[0, :, DURATION_COLUMN])
                print("----")
                print("pitch ", torch.argmax(pitch_preds[0], dim=-1))
                print("velocity ", torch.argmax(velocity_preds[0], dim=-1))
                print("step ", step_preds[0])
                print("duration ", duration_preds[0])
                print("loss ", loss.item())
                print("loss_pitch ", loss_pitch.item())
                print("loss_velocity ", loss_velocity.item())
                print("loss_step ", loss_step.item())
                print("loss_duration ", loss_duration.item())
                print()

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