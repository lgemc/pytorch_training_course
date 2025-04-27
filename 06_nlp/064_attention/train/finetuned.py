from typing import List
from transformers import GPT2LMHeadModel
import torch
from torch import nn
from torch.utils.data import DataLoader

def freeze_all_except_head(
        model,
):
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze final layer norm (ln_f)
    for param in model.transformer.ln_f.parameters():
        param.requires_grad = True

    # Unfreeze lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = True


def unfreeze_all_except_head(
        model
):
    for param in model.parameters():
        param.requires_grad = True

def train(
        model: GPT2LMHeadModel,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        logging_steps: int,
        vocab_size: int,
) -> (GPT2LMHeadModel, List, List):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.train()

    test_losses, train_losses = [], []

    for epoch in range(epochs):
        total_train_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % logging_steps == 0:
                print(f"Step {step}, Loss: {loss.item()}")

            total_train_loss += loss.item()

        train_losses.append(total_train_loss/len(train_dataloader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x, labels=y)
                val_loss += outputs.loss.item()


        val_loss /= len(val_dataloader)
        test_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")

    return model, train_losses, test_losses