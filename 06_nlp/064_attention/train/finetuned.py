from transformers import GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader

def train(
        model: GPT2LMHeadModel,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        logging_steps: int,
) -> GPT2LMHeadModel:
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_dataloader):
            x, _ = batch # gpt model already calculates labels, which are the same as input
            x = x.to(device)
            outputs = model(x, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % logging_steps == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                outputs = model(batch, labels=batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss}")

    return model