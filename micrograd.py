"""Karpathy's micrograd, reimplemented: scalar reverse-mode autograd.

This is the scalar ancestor of engine.py. A `Value` holds one number and
remembers the operation that produced it; `.backward()` walks the graph in
reverse and accumulates `.grad` via the chain rule. engine.py is the same idea
lifted to NumPy arrays (with broadcasting), which is the only real complication.

Credit: the design is Andrej Karpathy's micrograd. I rewrote it to understand
it line by line rather than to copy it.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # product rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only constant powers"
        out = Value(self.data**p, (self,), f"**{p}")

        def _backward():
            self.grad += p * self.data ** (p - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # build a reverse topological order, then push gradients along it
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # sugar
    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1 if isinstance(other, Value) else Value(other) ** -1)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == "__main__":
    # tiny expression, gradients worked out by hand so you can check them.
    # a=2, b=1:  c=2(a+b)+1=7 (relu active),  d=a*b+b^3=3,  f=relu(c)+d=10
    #   df/da = 2 (through relu) + 1 (through d) = 3
    #   df/db = 2 (through relu) + (a+3b^2)=5 (through d) = 7
    a = Value(2.0)
    b = Value(1.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    e = c.relu()
    f = e + d
    f.backward()
    print(f"f = {f.data:.4f}")  # 10.0000
    print(f"df/da = {a.grad:.4f}")  # 3.0000
    print(f"df/db = {b.grad:.4f}")  # 7.0000
