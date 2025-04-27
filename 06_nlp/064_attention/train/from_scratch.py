import torch
from torch import nn
from torch.utils.data import DataLoader


def train(
        model: nn.Module,
        vocab_size: int,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        logging_steps: int,
) -> nn.Module:
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # IMPORTANT: Pass the entire batch, not individual tokens
            outputs = model(x)  # Shape: [batch_size, block_size, vocab_size]

            # Reshape outputs and targets for loss calculation
            # outputs: [batch_size * block_size, vocab_size]
            # y: [batch_size * block_size]
            loss = criterion(outputs.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % logging_steps == 0:
                print(f"Step {step}, Loss: {loss.item()}, Total steps: {len(train_dataloader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                val_loss += criterion(outputs.view(-1, vocab_size), y.view(-1)).item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss}")
        model.train()  # Set back to training mode

    return model