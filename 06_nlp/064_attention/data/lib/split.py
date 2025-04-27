from torch.utils.data import Subset, Dataset

def train_test_split_sequential(
        dataset: Dataset,
        test_size: float = 0.2,
):
    """
    Splits a dataset into training and testing sets sequentially.
    :param dataset: The dataset to split.
    :param test_size: The proportion of the dataset to include in the test split.
    :return: train_dataset, test_dataset
    """
    n = len(dataset)
    test_len = int(n * test_size)
    train_len = n - test_len

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, n))

    return Subset(dataset, train_indices), Subset(dataset, test_indices)