MULTIPLICATION = '*'
SUM = '+'
REST = '-'

class BinaryOperation:
    def __init__(self, a: int, b: int, operation: str):
        self.a = a
        self.b = b
        self.operation = operation

def get_operation_result(operation: BinaryOperation) -> int:
    operator = operation.operation

    if operator == MULTIPLICATION:
            return operation.a * operation.b
    elif operator == SUM:
        return operation.a + operation.b
    elif operator == REST:
        return operation.a - operation.b