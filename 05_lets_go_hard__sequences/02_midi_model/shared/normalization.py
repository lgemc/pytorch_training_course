from typing import List, Tuple

def normalize(data: List) -> Tuple[float, float, List]:
    """
    Normalize the data to a range of 0 to 1.
    """
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
    return min_value, max_value, normalized_data