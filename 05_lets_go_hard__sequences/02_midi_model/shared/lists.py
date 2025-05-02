from typing import List, Tuple

def build_ngrams(elements: List, n=5) -> List[Tuple[List, any]]:
    ngrams = []
    for i in range(len(elements) - n):
        ngram = elements[i:i + n]
        ngrams.append((ngram, elements[i + n]))

    return ngrams