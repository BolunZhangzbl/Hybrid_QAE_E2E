# -- Public Imports
import gc
import os
import random
import pennylane as qml
from pennylane import numpy as np

import cirq
import timeit
import tensorflow_quantum as tfq

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Activation, Reshape, Flatten


# --Private Imports


# -- Global Variables


# -- Functions

def get_onehot_data(data_size, bit_num):
    # Generate random integer indices between 0 and num_bits-1
    num_bits_onehot = 2**bit_num
    random_indices = np.random.randint(0, num_bits_onehot, size=data_size)

    # Create an array of zeros with shape (data_size, num_bits)
    onehot_data = np.zeros((data_size, num_bits_onehot), dtype=int)

    # Set the appropriate indices to 1 for one-hot encoding
    onehot_data[np.arange(data_size), random_indices] = 1

    return onehot_data

def create_qml_circuit(dev, num_qubits, bit_num):
    def layer1(layer_weights):
        """
        q layer for fading:
        :param layer_weights:
        :return:
        """
        for wire in range(num_qubits):
            qml.RY(layer_weights[0, wire, 0] * np.pi, wires=wire)
        for wire in range(0, num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1)])

    @qml.qnode(dev, interface='tf', diff_method='best')
    def qcircuit_complex(inputs, weights):

        qml.templates.AmplitudeEmbedding(inputs, wires=range(bit_num), normalize=True)

        for layer_weights in weights:
            layer1(layer_weights)

        return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]

    return qcircuit_complex


# Define the quantum circuit for TFQ
def create_tfq_circuit(num_qubits, bit_num):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    inputs = tf.keras.Input(shape=(2 ** bit_num,))

    # Embed the data
    circuit += tfq.convert_to_tensor([cirq.Circuit(cirq.rx(np.pi * x)(qubits[i]) for i, x in enumerate(range(bit_num)))])

    # Add quantum layers
    def layer1_tfq(circuit, qubits, layer_weights):
        for i, qubit in enumerate(qubits):
            circuit += cirq.ry(layer_weights[0, i, 0] * np.pi)(qubit)
        for i in range(len(qubits) - 1):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])

    # Build the full circuit
    weights = tf.keras.Input(shape=(3, num_qubits, 1))
    for layer_weights in weights:
        layer1_tfq(circuit, qubits, layer_weights)

    # Define the model
    model = tfq.layers.PQC(circuit, operators=[cirq.Z(qubits[i]) for i in range(num_qubits)])
    return tf.keras.Model(inputs=[inputs, weights], outputs=model(inputs))


def transmitter(num_qubits, bit_num):
    ipl = Input((2**bit_num,))
    d1 = Dense(2**bit_num, activation='relu')(ipl)
    d2 = Dense(2*num_qubits, activation='linear')(d1)
    rl = Reshape((num_qubits, 2))(d2)
    nl = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * (K.l2_normalize(x, axis=[1, 2])))(rl)

    model = tf.keras.models.Model(inputs=ipl, outputs=nl)

    return model


def time_inference(iters=100):
    dev = qml.device("default.qubit", wires=4)
    weight_init = np.random.rand(3, 1, 4, 1)
    qml_circuit = create_qml_circuit(dev, 4, 4, complex=True)
    dnn = transmitter(4, 4)
    tfq_circuit = create_tfq_circuit(4, 4)

    data = get_onehot_data(32, 4)

    # Pennylane Circuit Timing
    print(f"1. Calculating avg running time for Q Circuit using Pennylane......")
    execution_time = timeit.timeit(lambda: qml_circuit(data, weight_init), number=iters)
    avg_time = execution_time / iters
    print(f"1. Finish calculating avg running time for Q Circuit using Pennylane - {avg_time} sec.")

    # DNN Timing
    print(f"2. Calculating avg running time for DNN......")
    execution_time = timeit.timeit(lambda: dnn(data), number=iters)
    avg_time = execution_time / iters
    print(f"2. Finish calculating avg running time for DNN - {avg_time} sec.")

    # TensorFlow Quantum Timing
    print(f"3. Calculating avg running time for Q Circuit using TensorFlow Quantum......")
    execution_time = timeit.timeit(lambda: tfq_circuit([data, weight_init]), number=iters)
    avg_time = execution_time / iters
    print(f"3. Finish calculating avg running time for Q Circuit using TensorFlow Quantum - {avg_time} sec.")


time_inference()