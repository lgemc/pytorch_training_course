from typing import List
from PIL import Image
import numpy as np

def print_image_as_char(arr: List[bytes], width: int, height: int) -> None:
    for row in range(height):
        for col in range(width):
            print('#' if arr[width * row + col] > 0 else '@', end='')

        print()

def convert_and_show_image(arr: List[bytes], width: int, height: int) -> None:
    image_array = np.frombuffer(arr, dtype=np.uint8).reshape((height, width))

    image = Image.fromarray(image_array, mode='L')

    image.show()