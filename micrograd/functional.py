from .engine import Value


def binary_cross_entropy(input: Value, target: int):
    return -(target * input.log() + (1 - target) * (1 - input).log())
