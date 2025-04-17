import torch

def print_image_as_char(arr: torch.Tensor , width: int, height: int) -> None:
    for row in range(height):
        for col in range(width):
            print('#' if arr[width * row + col] > 0 else '@', end='')

        print()