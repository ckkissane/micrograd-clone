import random
from engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


def matmul(A, B):
    # both A and B must be 2d lists
    assert len(A[0]) == len(B)

    out = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(out)):
        for j in range(len(out[0])):
            for k in range(len(B)):
                out[i][j] += A[i][k] * B[k][j]

    return out


class Linear(Module):
    def __init__(self, nin, nout):
        self.nin, self.nout = nin, nout
        self.weight = [
            [Value(random.uniform(-1, 1)) for _ in range(nout)] for _ in range(nin)
        ]
        self.bias = [Value(0.0) for _ in range(nout)]

    def __call__(self, x):
        # x must be 1D list
        # y = xW + b (note: pytorch does xW^T + b)
        xW = matmul([x], self.weight)
        y = [xwi + bi for xwi, bi in zip(xW[0], self.bias)]
        return y

    def parameters(self):
        return [wp for row in self.weight for wp in row] + [bp for bp in self.bias]

    def __repr__(self):
        return f"Linear(nin={self.nin}, nout={self.nout})"


class Sigmoid(Module):
    def __call__(self, x):
        # x must be 1D list
        return [xi.sigmoid() for xi in x]

    def __repr__(self):
        return "Sigmoid()"
