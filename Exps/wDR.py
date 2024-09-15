import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

batch_size=  32

num_qubits = 4   # number of wires
num_layers = 1
blocks = 4
weights_init = 0.01 * np.random.rand(blocks, num_layers, num_qubits, 3, requires_grad=True)
inputs_tensor = np.random.rand(batch_size, 2)

dev = qml.device("default.qubit", wires=num_qubits)


def angle_preparation(inputs, layer_weights):
    weighted_inputs = np.zeros((32, num_qubits))
    for ind in range(num_qubits):
        weighted_inputs[:, ind] = inputs[:, ind % 2] * layer_weights[0, ind, 0]
    qml.templates.AngleEmbedding(weighted_inputs, wires=range(num_qubits))


@qml.qnode(dev)
def circuit(inputs, weights):
    for layer_weights in weights:
        angle_preparation(inputs, layer_weights)
        StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


qml.draw_mpl(circuit, show_all_wires=True)(inputs_tensor, weights_init)