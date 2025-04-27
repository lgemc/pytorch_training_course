import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import GPT2Tokenizer


def generate_text(
        model,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = "cpu"
) -> str:
    """
    Generate text from a trained language model.

    Args:
        model: The trained language model
        tokenizer: GPT2Tokenizer instance
        prompt: The starting text to generate from
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: If set, only sample from the top k most probable tokens
        top_p: If set, use nucleus sampling with this probability threshold
        device: Device to run the generation on

    Returns:
        Generated text as a string
    """
    model.eval()
    model.to(device)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    # Get the model's block size (maximum sequence length)
    block_size = model.config.n_positions

    # Generate tokens one by one
    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate the input if it exceeds the block size
            current_input = input_tensor[:, -block_size:]

            # Get model predictions
            out = model(current_input)

            # Get the logits
            logits = out.logits

            # Focus on the last token's predictions
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < values[:, -1]] = float('-inf')

            # Apply top-p (nucleus) filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Add the new token to our sequence
            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
            generated_ids.append(next_token_id.item())

            # Stop if we generate the end-of-sequence token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids)
    return generated_text