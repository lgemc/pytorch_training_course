from binary_operations import BinaryOperation, MULTIPLICATION, SUM, REST, get_operation_result

def generate_operations(num_operations: int):
    """
    Generates a list of binary operations with random integers and operations.

    Args:
        num_operations (int): The number of operations to generate.

    Returns:
        List[BinaryOperation]: A list of BinaryOperation objects.
    """
    import random

    operations = []
    dedup = set()
    for _ in range(num_operations):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        operation = random.choice([MULTIPLICATION, SUM, REST])

        # Check for duplicates
        if (a, b, operation) not in dedup:
            dedup.add((a, b, operation))
            operations.append(BinaryOperation(a, b, operation))

    return operations

def generate_dataset_file(
        num_operations: int = 10000,
        filename: str = "dataset.txt"
):
    """
    Generates a dataset file with binary operations.

    Args:
        num_operations (int): The number of operations to generate.
        filename (str): The name of the output file.
    """
    operations = generate_operations(num_operations)

    with open(filename, "w") as f:
        for op in operations:
            result = get_operation_result(op)
            if result < 0 or result > 9:
                continue
            f.write(f"{op.a} {op.operation} {op.b} = {result}\n")


if __name__ == "__main__":
    generate_dataset_file(num_operations=10000, filename="dataset.txt")