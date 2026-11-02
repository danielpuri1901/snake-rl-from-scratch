"""The policy network (cRL: model.c): 76 inputs -> 128 -> 128 -> 5 action probabilities."""
import numpy as np

from autograd import (
    build_graph, var_add, var_create, var_matmul, var_relu, var_reinforce_loss, var_softmax,
)


class ActorModel:
    """cRL: create_actor_model + model_state."""

    def __init__(self, rng):
        self.input = var_create(76, 1)

        W0, b0 = var_create(128, 76, parameter=True), var_create(128, 1, parameter=True)
        W1, b1 = var_create(128, 128, parameter=True), var_create(128, 1, parameter=True)
        W2, b2 = var_create(5, 128, parameter=True), var_create(5, 1, parameter=True)

        # Xavier-uniform init, same bounds as cRL. Biases start at 0.
        for W, fan_in, fan_out in [(W0, 76, 128), (W1, 128, 128), (W2, 128, 5)]:
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            W.val[:] = rng.uniform(-bound, bound, W.val.shape)

        z0 = var_add(var_matmul(W0, self.input), b0)
        a0 = var_relu(z0)

        z1 = var_add(var_matmul(W1, a0), b1)
        a1 = var_relu(z1)

        z2 = var_add(var_matmul(W2, a1), b2)
        self.output = var_softmax(z2)

        self.advantage = var_create(5, 1)
        self.cost = var_reinforce_loss(self.output, self.advantage)

        self.forward_graph = build_graph(self.output)
        self.cost_graph = build_graph(self.cost)
        self.parameters = [v for v in self.cost_graph if v.parameter]
