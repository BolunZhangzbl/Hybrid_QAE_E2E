# -- Public Imports
import pennylane as qml
from pennylane import numpy as np

import timeit
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Activation, Reshape, Flatten
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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


def create_qml_circuit(dev, num_qubits, bit_num, interface):
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

    @qml.qnode(dev, interface=interface, diff_method='best')
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
    # nl = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * (K.l2_normalize(x, axis=[1])))(d2)

    model = tf.keras.models.Model(inputs=ipl, outputs=d2)

    return model


def time_inference(iters=1000):
    dev_default = qml.device("default.qubit", wires=4)
    dev_lightning = qml.device("lightning.gpu", wires=4)
    weight_qml = np.random.rand(3, 1, 4, 1) * 0.01
    qml_circuit_default0 = create_qml_circuit(dev_default, 4, 4, 'tensorflow')
    qml_circuit_default1 = create_qml_circuit(dev_default, 4, 4, None)
    qml_circuit_lightning0 = create_qml_circuit(dev_lightning, 4, 4, 'tensorflow')
    qml_circuit_lightning1 = create_qml_circuit(dev_lightning, 4, 4, None)

    weight_tfq = np.random.rand(3, 1, 4, 1)
    # tfq_circuit = create_tfq_circuit(num_qubits=4, bit_num=4)

    weight_qiskit = np.random.rand(3, 1, 4, 1)

    dnn = transmitter(4, 4)

    data = get_onehot_data(1, 4)[0]
    # data = np.random.rand(1024, 4)

    # Pennylane Circuit Timing - Default
    print(f"\n0. Calculating avg running time for Q Circuit using Pennylane - Default......")
    execution_time = timeit.timeit(lambda: qml_circuit_default0(data, weight_qml), number=iters)
    avg_time0 = execution_time / iters
    print(f"0. Finish calculating avg running time for Q Circuit using Pennylane - Default - {avg_time0} sec.")

    # Pennylane Circuit Timing - Default
    print(f"\n1. Calculating avg running time for Q Circuit using Pennylane - Default......")
    execution_time = timeit.timeit(lambda: qml_circuit_default1(data, weight_qml), number=iters)
    avg_time1 = execution_time / iters
    print(f"1. Finish calculating avg running time for Q Circuit using Pennylane - Default - {avg_time1} sec.")

    # Pennylane Circuit Timing - Lightning
    print(f"\n2. Calculating avg running time for Q Circuit using Pennylane - Lightning......")
    execution_time = timeit.timeit(lambda: qml_circuit_lightning0(data, weight_qml), number=iters)
    avg_time2 = execution_time / iters
    print(f"2. Finish calculating avg running time for Q Circuit using Pennylane - Lightning - {avg_time2} sec.")

    # Pennylane Circuit Timing - Lightning
    print(f"\n3. Calculating avg running time for Q Circuit using Pennylane - Lightning......")
    execution_time = timeit.timeit(lambda: qml_circuit_lightning1(data, weight_qml), number=iters)
    avg_time3 = execution_time / iters
    print(f"3. Finish calculating avg running time for Q Circuit using Pennylane - Lightning - {avg_time3} sec.")

    # DNN Timing
    print(f"\n4. Calculating avg running time for DNN......")
    execution_time = timeit.timeit(lambda: dnn(np.expand_dims(data, axis=0)), number=iters)
    avg_time4 = execution_time / iters
    print(f"4. Finish calculating avg running time for DNN - {avg_time4} sec.")

    # Generating the bar plot for avg_time comparisons
    methods = ['Pennylane (default, tf)', 'Pennylane (default, None)',
               'Pennylane (lightning, tf)', 'Pennylane (lightning, None)',
               'Tensorflow']
    avg_times = [avg_time0, avg_time1, avg_time2, avg_time3, avg_time4]

    plt.figure(figsize=(15, 10))
    plt.bar(methods, avg_times, color=['blue', 'green', 'red', 'orange', 'purple'])
    plt.ylabel('Average Time (sec)')
    plt.title('Average Running Time Comparison for Different Quantum and DNN Methods')
    plt.show()


time_inference()