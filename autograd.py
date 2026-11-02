"""Tiny autograd engine over numpy matrices (cRL: autograd.c + mat.c)."""
import numpy as np


class Var:
    """One node in the graph (cRL: struct Var)."""

    def __init__(self, val, op=None, inputs=(), requires_grad=False, parameter=False):
        self.val = val                      # numpy matrix (cRL: matrix* val)
        self.grad = None if val is None else np.zeros_like(val)  # cRL: matrix* grad
        self.op = op                        # how it was made (cRL: const VarType* type)
        self.inputs = inputs                # Vars it was made from (cRL: Var* inputs[2])
        self.requires_grad = requires_grad  # cRL: VAR_FLAG_REQUIRES_GRAD
        self.parameter = parameter          # cRL: VAR_FLAG_PARAMETER


class VarType:
    """One kind of op: its forward and backward functions (cRL: struct VarType)."""

    def __init__(self, name, forward, backward):
        self.name = name
        self.forward = forward
        self.backward = backward


def var_create(rows, cols, parameter=False):
    """A leaf Var: input data, or a weight/bias when parameter=True."""
    return Var(np.zeros((rows, cols), np.float32), requires_grad=parameter, parameter=parameter)

# Columns are samples, so the same code runs one sample (shape (n, 1), like cRL)
# or a batch of samples (shape (n, batch)).

def _relu_forward(v):
    v.val = np.maximum(v.inputs[0].val, 0)


def _relu_backward(v):
    a = v.inputs[0]
    if a.requires_grad:
        a.grad += (a.val > 0) * v.grad


def _softmax_forward(v):
    z = v.inputs[0].val
    e = np.exp(z - z.max(axis=0, keepdims=True))
    v.val = e / e.sum(axis=0, keepdims=True)


def _softmax_backward(v):
    a = v.inputs[0]
    if a.requires_grad:
        dot = (v.grad * v.val).sum(axis=0, keepdims=True)
        a.grad += v.val * (v.grad - dot)


def _add_forward(v):
    v.val = v.inputs[0].val + v.inputs[1].val


def _add_backward(v):
    for x in v.inputs:
        if x.requires_grad:
            # A bias of shape (n, 1) is added to every column, so its grad sums over columns.
            x.grad += v.grad if v.grad.shape == x.grad.shape else v.grad.sum(axis=1, keepdims=True)


def _matmul_forward(v):
    v.val = v.inputs[0].val @ v.inputs[1].val


def _matmul_backward(v):
    a, b = v.inputs
    if a.requires_grad:
        a.grad += v.grad @ b.val.T
    if b.requires_grad:
        b.grad += a.val.T @ v.grad


def _reinforce_forward(v):
    probs, advantages = v.inputs
    v.val = -np.log(np.maximum(probs.val, 1e-8)) * advantages.val


def _reinforce_backward(v):
    probs, advantages = v.inputs
    if probs.requires_grad:
        probs.grad += v.grad * (-advantages.val / np.maximum(probs.val, 1e-8))

VAR_TYPE_RELU = VarType("relu", _relu_forward, _relu_backward)
VAR_TYPE_SOFTMAX = VarType("softmax", _softmax_forward, _softmax_backward)
VAR_TYPE_ADD = VarType("add", _add_forward, _add_backward)
VAR_TYPE_MATMUL = VarType("matmul", _matmul_forward, _matmul_backward)
VAR_TYPE_REINFORCE_LOSS = VarType("reinforce_loss", _reinforce_forward, _reinforce_backward)


def _create_node(op, *inputs):
    # Building a node does no math yet (cRL: create_node). forward_pass fills val later.
    return Var(None, op, inputs, requires_grad=any(x.requires_grad for x in inputs))


def var_relu(a):
    return _create_node(VAR_TYPE_RELU, a)


def var_softmax(a):
    return _create_node(VAR_TYPE_SOFTMAX, a)


def var_add(a, b):
    return _create_node(VAR_TYPE_ADD, a, b)


def var_matmul(a, b):
    return _create_node(VAR_TYPE_MATMUL, a, b)


def var_reinforce_loss(probs, advantages):
    return _create_node(VAR_TYPE_REINFORCE_LOSS, probs, advantages)

def build_graph(out):
    """Every Var that `out` depends on, inputs before outputs (cRL: build_graph)."""
    order, seen = [], set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for x in v.inputs:
            visit(x)
        order.append(v)

    visit(out)
    return order


def forward_pass(graph):
    for v in graph:
        if v.op is not None:
            v.op.forward(v)


def backward_pass(graph):
    # Parameter grads are not cleared here: they add up until the trainer clears them.
    for v in graph:
        if v.requires_grad and not v.parameter:
            v.grad = np.zeros_like(v.val)
    graph[-1].grad = np.ones_like(graph[-1].val)
    for v in reversed(graph):
        if v.requires_grad and v.op is not None:
            v.op.backward(v)


if __name__ == "__main__":
    # Gradient check: autograd grads must match finite differences on a small 2-layer policy.
    rng = np.random.default_rng(0)

    def leaf(rows, cols, parameter=False):
        return Var(rng.normal(size=(rows, cols)), requires_grad=parameter, parameter=parameter)

    x, adv = leaf(4, 3), leaf(3, 3)
    W0, b0, W1, b1 = leaf(5, 4, True), leaf(5, 1, True), leaf(3, 5, True), leaf(3, 1, True)
    h = var_relu(var_add(var_matmul(W0, x), b0))
    cost = var_reinforce_loss(var_softmax(var_add(var_matmul(W1, h), b1)), adv)
    graph = build_graph(cost)

    forward_pass(graph)
    backward_pass(graph)

    eps = 1e-6
    for name, p in [("W0", W0), ("b0", b0), ("W1", W1), ("b1", b1)]:
        numeric = np.zeros_like(p.val)
        for i in np.ndindex(p.val.shape):
            old = p.val[i]
            p.val[i] = old + eps
            forward_pass(graph)
            plus = cost.val.sum()
            p.val[i] = old - eps
            forward_pass(graph)
            minus = cost.val.sum()
            p.val[i] = old
            numeric[i] = (plus - minus) / (2 * eps)
        err = np.abs(numeric - p.grad).max() / max(np.abs(numeric).max(), 1e-12)
        print(f"{name}: max relative error {err:.2e}")
        assert err < 1e-6, f"gradient mismatch in {name}"
    print("gradient check passed")