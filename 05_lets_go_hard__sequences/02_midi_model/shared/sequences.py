from typing import List, Tuple

def build(elements: List, n=5) -> List[Tuple[List, List]]:
    """
    Build sequences of n elements from a list, [(input, targets)] where targets ar inputs shifted to the right.
    """
    sequences = []
    for i in range(len(elements) - n):
        input = elements[i:i + n]
        target = elements[i + 1:i + n + 1]
        sequences.append((input, target))

    return sequences