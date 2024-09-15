import pennylane as qml
from pennylane import numpy as np

# Initialize a device with the 'default.qubit' backend
num_qubits = 4  # Change this to the desired number of qubits
dev = qml.device('default.qubit', wires=num_qubits)


# 1. Hadamard Circuit
@qml.qnode(dev)
def hadamard_circuit(features):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    return qml.state()


# 2. BasisEmbedding Circuit
@qml.qnode(dev)
def basis_embedding_circuit(features):
    qml.BasisEmbedding(features, wires=range(num_qubits))
    return qml.state()


# 3. BasisState Circuit
@qml.qnode(dev)
def basis_state_circuit(state):
    qml.BasisState(np.array(state), wires=range(num_qubits))
    return qml.state()


# 4. AngleEmbedding Circuit
@qml.qnode(dev)
def angle_embedding_circuit(features):
    qml.AngleEmbedding(features, wires=range(num_qubits), rotation='X')
    return qml.state()


# 5. AmplitudeEmbedding Circuit
@qml.qnode(dev)
def amplitude_embedding_circuit(features):
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize=True)
    return qml.state()


# 6. MottonenStatePreparation Circuit
@qml.qnode(dev)
def mottonen_sp_circuit(features):
    qml.MottonenStatePreparation(features, wires=range(num_qubits))
    return qml.state()


# Example usage
# 1. Hadamard
features = [1, 1, 1, 0]  # Binary features to embed
print("\nHadamard Circuit State:")
print(hadamard_circuit(features))
print(hadamard_circuit(features).shape)

# 2. BasisEmbedding
features = [1, 0, 1, 0]  # Binary features to embed
print("\nBasisEmbedding Circuit State:")
print(basis_embedding_circuit(features))
print(basis_embedding_circuit(features).shape)

# 3. BasisState
state = [1, 0, 1, 0]  # Basis state to prepare
print("\nBasisState Circuit State:")
print(basis_state_circuit(state))
print(basis_state_circuit(state).shape)

# 4. AngleEmbedding
features = [np.pi / 2, np.pi / 4, np.pi / 6, np.pi / 8]  # Angles for embedding
print("\nAngleEmbedding Circuit State:")
print(angle_embedding_circuit(features))
print(angle_embedding_circuit(features).shape)

# 5. AmplitudeEmbedding
features = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Amplitudes for embedding
print("\nAmplitudeEmbedding Circuit State:")
print(amplitude_embedding_circuit(features))
print(amplitude_embedding_circuit(features).shape)

# 6. MottonenStatePreparation Circuit
features = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Amplitudes for embedding
print("\nMottonenStatePreparation Circuit State:")
print(mottonen_sp_circuit(features))
print(mottonen_sp_circuit(features).shape)