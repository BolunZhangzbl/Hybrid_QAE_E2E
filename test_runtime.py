# -- Public Imports
import gc
import os
import random
import pennylane as qml
from pennylane import numpy as np

# import cirq
import timeit

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

    @qml.qnode(dev, diff_method='adjoint', mutable=False)
    def qcircuit_complex(inputs, weights):

        qml.AmplitudeEmbedding(inputs, wires=range(bit_num), normalize=True)

        for layer_weights in weights:
            layer1(layer_weights)

        return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]

    return qcircuit_complex


def transmitter(num_qubits, bit_num):
    ipl = Input((2**bit_num,))
    d1 = Dense(2**bit_num, activation='relu')(ipl)
    d2 = Dense(num_qubits, activation='linear')(d1)
    nl = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * (K.l2_normalize(x, axis=[1])))(d2)

    model = tf.keras.models.Model(inputs=ipl, outputs=nl)

    return model


def time_inference(iters=1000):
    dev = qml.device("lightning.qubit", wires=4)
    weight_init = np.random.rand(3, 1, 4, 1)
    qml_circuit = create_qml_circuit(dev, 4, 4)
    dnn = transmitter(4, 4)
    # tfq_circuit = create_tfq_circuit(4, 4)

    data = get_onehot_data(32, 4)

    # Pennylane Circuit Timing
    print(f"0. Calculating avg running time for Q Circuit using Pennylane......")
    execution_time = timeit.timeit(lambda: qml_circuit(data, weight_init), number=iters)
    avg_time0 = execution_time / iters
    print(f"0. Finish calculating avg running time for Q Circuit using Pennylane - {avg_time0} sec.")

    # DNN Timing
    print(f"1. Calculating avg running time for DNN......")
    execution_time = timeit.timeit(lambda: dnn(data), number=iters)
    avg_time1 = execution_time / iters
    print(f"1. Finish calculating avg running time for DNN - {avg_time1} sec.")


time_inference()