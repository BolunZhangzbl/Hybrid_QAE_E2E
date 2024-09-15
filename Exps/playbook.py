import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

import matplotlib.pyplot as plt



num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)
blocks = 2
num_layers = 1

weights_init = np.full((blocks, num_layers, num_qubits, 3), 0.5)
inputs_init = np.array([0.8, 0.2])
# inputs_init = np.array([9.43241289e-05, 4.99622298e-03])
# weights_init = 0.01 * np.random.rand(blocks, num_layers, num_qubits, 3, requires_grad=True)
# inputs_init = np.random.rand(num_qubits)
# -- Functions


def reuploading_layer(inputs, layer_weights, idx):
    print(f"Inputs of {idx} layer: ")
    print(inputs)

    qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.Snapshot(f"Output of the {idx} AngleEmbedding layer")
    # StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))


@qml.qnode(dev, diff_method='best')
def circuit(inputs, weights):

    for idx, layer_weights in enumerate(weights):
        reuploading_layer(inputs, layer_weights, idx)
        qml.Snapshot(f"Output of the {idx} layer")

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


results = qml.snapshots(circuit)(inputs_init, weights_init)


# print(f"inputs_init: {inputs_init}\n")
# print(f"weights_init: {weights_init}\n\n")
for key, val in results.items():
    print(f"\n{key}: {val}")
    print(len(val), "\n")