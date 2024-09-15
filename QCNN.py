# -- Public Imports
import gc
import os
import random
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Conv1D, Activation, BatchNormalization, Reshape, Flatten

# --Private Imports

# -- Global Variables


bit_num = 4
num_qubits = 4  # number of wires
num_wires = 2
R = float(bit_num / num_qubits)
dev = qml.device("default.qubit", wires=num_wires)

channel_type = 'rayleigh'

batch_size = 32
snr_train = 10

num_layers = 2
weights_init = 0.01 * np.random.rand(num_layers, 1, num_wires, 3, requires_grad=True)
inputs_weights_init = 0.01 * np.random.rand(num_layers, num_wires, requires_grad=True)


# -- Functions

########################################  Utils function for layers  ########################################


def layer1(layer_weights):
    # Apply rotational gates
    for wire in range(num_wires):
        qml.U3(*layer_weights[0, wire], wires=wire)

    for wire in range(num_wires):
        qml.CZ(wires=[wire, (wire + 1) % num_wires])


def weighted_reuploading_layer(inputs, layer_weights):
    weighted_inputs = inputs * layer_weights[0, :, 0]  # (32,num_wires) * (num_wires, )

    qml.templates.AngleEmbedding(weighted_inputs, wires=range(num_wires), rotation='Y')
    layer1(layer_weights)


def weighted_reuploading_layer_new(inputs, layer_weights, layer_inputs_weights):
    weighted_inputs = inputs * layer_inputs_weights  # (32, num_wires) * (num_wires, )

    qml.templates.AngleEmbedding(weighted_inputs, wires=range(num_wires), rotation='Y')
    layer1(layer_weights)


########################################  Function for circuits  ########################################

@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit11(inputs, weights):
    qml.templates.AngleEmbedding(inputs * np.pi, wires=range(num_wires), rotation='Y')

    for layer_weights in weights:
        layer1(layer_weights)

    return qml.expval(qml.PauliZ(num_wires - 1))


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit12(inputs, weights):
    qml.templates.AngleEmbedding(inputs * np.pi, wires=range(num_wires), rotation='Y')
    for layer_weights in weights:
        layer1(layer_weights)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_wires)]


# (Weighted) Data Re-uploading Methods

@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit31(inputs, weights):
    for layer_weights in weights:
        qml.templates.AngleEmbedding(inputs, wires=range(num_wires), rotation='Y')
        StronglyEntanglingLayers(layer_weights, wires=range(num_wires))

    return qml.expval(qml.PauliZ(num_wires - 1))


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit32(inputs, weights):
    for layer_weights in weights:
        qml.templates.AngleEmbedding(inputs, wires=range(num_wires), rotation='Y')
        layer1(layer_weights)

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_wires)]


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit41(inputs, weights):
    for layer_weights in weights:
        weighted_reuploading_layer(inputs, layer_weights)

    return qml.expval(qml.PauliZ(num_wires - 1))


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit42(inputs, weights):
    for layer_weights in weights:
        weighted_reuploading_layer(inputs, layer_weights)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_wires)]


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit51(inputs, weights, inputs_weights):
    for layer_weights, layer_inputs_weights in zip(weights, inputs_weights):
        weighted_reuploading_layer_new(inputs, layer_weights, layer_inputs_weights)

    return qml.expval(qml.PauliZ(num_wires - 1))


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit52(inputs, weights, inputs_weights):
    for layer_weights, layer_inputs_weights in zip(weights, inputs_weights):
        weighted_reuploading_layer_new(inputs, layer_weights, layer_inputs_weights)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_wires)]


# New Circuit (Original was from Receiver)

@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit6(inputs, weights):
    # qml.templates.AmplitudeEmbedding(inputs, wires=range(num_wires), normalize=True)
    qml.templates.AngleEmbedding(inputs * np.pi, wires=range(num_wires))

    for layer_weights in weights[:-1]:
        layer1(layer_weights)

    for i in range(num_wires):
        qml.U3(*weights[-1, 0, i], wires=i)

    for i in range(num_wires - 2):
        qml.CZ(wires=[i, i + 2])

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_wires)]


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit71(inputs, weights):
    for wire in range(num_wires):
        qml.Hadamard(wires=wire)
    qml.templates.AngleEmbedding(inputs, wires=range(num_wires), rotation='Y')

    for layer_weights in weights:
        for wire in range(num_wires):
            qml.CRZ(layer_weights[0, wire, 0], wires=[wire, (wire + 1) % num_wires])
        for wire in range(num_wires):
            qml.RY(layer_weights[0, wire, 1], wires=wire)

    return qml.expval(qml.PauliZ(num_wires - 1))


@qml.qnode(dev, interface='tf', diff_method='backprop')
def circuit72(inputs, weights):
    for wire in range(num_wires):
        qml.Hadamard(wires=wire)
    qml.templates.AngleEmbedding(inputs, wires=range(num_wires), rotation='Y')

    for layer_weights in weights:
        for wire in range(num_wires):
            qml.CRZ(layer_weights[0, wire, 0], wires=[wire, (wire + 1) % num_wires])
        for wire in range(num_wires):
            qml.RY(layer_weights[0, wire, 1], wires=wire)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_wires)]


########################################  Wrapper Function for circuits  ########################################
dict_circuit = {'11': circuit11, '12': circuit12, '31': circuit31, '32': circuit32,
                '41': circuit41, '42': circuit42, '51': circuit51, '52': circuit52,
                '71': circuit71, '72': circuit72}


########################################  Callback Utils Func/Class  ##########################################

# Define the custom learning rate scheduler callback
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 10:
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        elif epoch % 10 == 0:
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate) * tf.math.exp(-0.1)
        else:
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"Epoch {epoch + 1}: Learning rate is {lr:.5f}.")


########################################  QAE, Channel  ########################################


def AWGN_Channel_tf(x, noise_stddev):
    noise = K.random_normal(K.shape(x), mean=0, stddev=noise_stddev)
    results = Add()([x, noise])

    return results


def Rayleigh_Channel_tf(x, noise_stddev):
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]
    H_R = K.random_normal(K.shape(x_real), mean=0, stddev=tf.sqrt(1 / 2))
    H_I = K.random_normal(K.shape(x_imag), mean=0, stddev=tf.sqrt(1 / 2))

    y_real = H_R * x_real - H_I * x_imag
    y_imag = H_R * x_imag + H_I * x_real

    noise_r = K.random_normal(K.shape(y_real), mean=0, stddev=noise_stddev)
    noise_i = K.random_normal(K.shape(y_imag), mean=0, stddev=noise_stddev)

    y_real = Add()([y_real, noise_r])
    y_imag = Add()([y_imag, noise_i])

    # Concatenating the real and imaginary components of received signal
    y = K.stack([y_real, y_imag], axis=-1)

    # Get the perfect channel response
    h_perfect = K.stack([H_R, H_I], axis=-1)

    # Concatenate the impaired signal and the estimated channel response

    return K.concatenate([y, h_perfect], axis=-1)


class Channel_Model(tf.keras.Model):
    """"""

    def __init__(self, channel_type=None, noise_stddev=None):
        super(Channel_Model, self).__init__()
        assert channel_type in ('awgn', 'rayleigh')

        self.channel_type = 'awgn' if channel_type is None else channel_type
        self.noise_stddev = np.sqrt(1 / (2 * R * (10 ** (7 / 10.0)))) if noise_stddev is None else noise_stddev

    def call(self, x):

        if self.channel_type == 'awgn':
            return AWGN_Channel_tf(x, self.noise_stddev)
        else:
            return Rayleigh_Channel_tf(x, self.noise_stddev)


########################################  Save Models  ##########################################


def save_history(history, ae_type='qcnn', num_qubits=num_qubits, bit_num=bit_num, snr_train=7, save=False):
    assert ae_type in ('qcnn', 'cnn')

    history_dict = history.history if hasattr(history, 'history') else history
    if save:
        file_path = f"{num_qubits}{bit_num}_qcnn/history/history_{ae_type}_{num_qubits}{bit_num}_SNR{snr_train}.npz"
        # file_path = os.path.join(os.getcwd(), file_path)
        np.savez(file_path, **history_dict)

    print(f"Successfully saved history dict for {ae_type} {num_qubits}{bit_num}!!!")


########################################  Utils Func to save & read  ##########################################

def create_folders(num_qubits=num_qubits, bit_num=bit_num):
    dirs = [f"{num_qubits}{bit_num}_qcnn/history",
            f"{num_qubits}{bit_num}_qcnn/models",
            f"{num_qubits}{bit_num}_qcnn/lists"]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory: {dir}!")
    return


########################################  Loss Func  ##########################################


def get_bler(x_in, x_out):
    assert len(x_in) == len(x_out)
    no_errors = 0
    datasize = len(x_in)
    for m in range(datasize):
        if not np.allclose(x_in[m], x_out[m]):
            no_errors += 1

    bler_val = no_errors / datasize
    return bler_val


def get_ber(x_in, x_out):
    assert len(x_in) == len(x_out)
    ber_val = np.mean((x_in != x_out).astype(int))

    return ber_val


########################################  Metric Func  ##########################################


def onehot_to_binary(onehot_tensor):
    # Convert onehot tensor to integer tensor
    int_tensor = tf.expand_dims(tf.argmax(onehot_tensor, axis=1, output_type=tf.int32), axis=1)

    num_classes = tf.shape(onehot_tensor)[1]
    num_bits = tf.cast(tf.math.ceil(tf.math.log(tf.cast(num_classes, tf.float32)) / tf.math.log(2.0)), tf.int32)

    bit_positions = tf.range(num_bits)

    binary_tensor = tf.bitwise.bitwise_and(tf.bitwise.right_shift(int_tensor, bit_positions), 1)
    return tf.reverse(binary_tensor, axis=[1])


# @tf.keras.saving.register_keras_serializable()
def ber_metric(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    return tf.reduce_mean(tf.cast(tf.not_equal(tf.math.round(y_pred), y_true), tf.float32))


def ber_onehot(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    y_true_binary = onehot_to_binary(y_true)
    y_pred_binary = onehot_to_binary(y_pred)

    return tf.reduce_mean(tf.cast(tf.not_equal(y_pred_binary, y_true_binary), tf.float32))


########################################  Get Dataset  ##########################################

def get_binary_data(data_size, bit_num=bit_num):
    """"""
    data = np.random.binomial(1, 0.5, size=(data_size, bit_num))
    return np.expand_dims(data, axis=-1)


def get_onehot_data(data_size, bit_num=bit_num):
    # Generate random integer indices between 0 and num_bits-1
    num_bits_onehot = 2 ** bit_num
    random_indices = np.random.randint(0, num_bits_onehot, size=data_size)

    # Create an array of zeros with shape (data_size, num_bits)
    onehot_data = np.zeros((data_size, num_bits_onehot), dtype=int)

    # Set the appropriate indices to 1 for one-hot encoding
    onehot_data[np.arange(data_size), random_indices] = 1

    return onehot_data


########################################  Set Seeds  ##########################################

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def set_global_var(num_qubits_val, bit_num_val, use_onehot_val, num_epochs_val=100,
                   snr_train_val=10, batch_size_val=32, channel_type_val='rayleigh'):
    """"""
    global num_qubits, bit_num, use_onehot, num_epochs, snr_train, batch_size, R, channel_type

    num_qubits = num_qubits_val
    bit_num = bit_num_val
    use_onehot = use_onehot_val
    num_epochs = num_epochs_val
    snr_train = snr_train_val
    batch_size = batch_size_val
    channel_type = channel_type_val
    R = float(bit_num / num_qubits)
    print("R: ", R)
    print(channel_type)


########################################  Conv1D Block  ##########################################

class SwishLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SwishLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


tf.keras.utils.get_custom_objects().update({'SwishLayer': SwishLayer})


class MishLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MishLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.nn.tanh(tf.math.softplus(inputs))


tf.keras.utils.get_custom_objects().update({'MishLayer': MishLayer})


class Conv1DBlock(tf.keras.layers.Layer):
    """"""

    def __init__(self, filters=2, kernel_size=5, strides=1, padding='same',
                 activation='elu', name=None, **kwargs):
        super(Conv1DBlock, self).__init__(name=name, **kwargs)
        assert activation in ('elu', 'swish', 'mish')
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        self.conv1 = Conv1D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)
        if self.activation == 'elu':
            self.act = Activation('elu')
        elif self.activation == 'swish':
            self.act = SwishLayer()
        else:
            self.act = MishLayer()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


# We do not use this transmitter for Hybrid QCNN-CNN Autoencoder
def transmitter(use_onehot=False, num_qubits=num_qubits, bit_num=bit_num):
    """"""
    if use_onehot:
        num_dense = 2 ** bit_num
    else:
        num_dense = bit_num

    input_layer = Input((num_dense, 1))
    # conv_layer1 = Conv1DBlock(filters=num_dense*2, kernel_size=5, strides=1, padding='same', activation='elu')(input_layer)
    conv_layer2 = Conv1DBlock(filters=num_dense, kernel_size=5, strides=1, padding='same', activation='elu')(
        input_layer)
    conv_layer3 = Conv1DBlock(filters=num_dense // 2, kernel_size=3, strides=1, padding='same', activation='elu')(
        conv_layer2)
    conv_layer4 = Conv1D(filters=2 // 1, kernel_size=3, strides=1, padding='same')(conv_layer3)
    norm_layer = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * K.l2_normalize(x, axis=1),
                        name='norm_layer')(conv_layer4)

    model = tf.keras.models.Model(inputs=input_layer, outputs=norm_layer)

    return model


def receiver(use_onehot=False, num_qubits=num_qubits, bit_num=bit_num):
    """"""
    if use_onehot:
        num_dense = 2 ** bit_num
        activation = 'softmax'
    else:
        num_dense = bit_num
        activation = 'sigmoid'

    input_layer = Input((num_dense, 2 * 2))
    conv_layer5 = Conv1DBlock(filters=num_dense * 2, kernel_size=5, strides=1, padding='same', activation='elu')(
        input_layer)
    conv_layer6 = Conv1DBlock(filters=num_dense, kernel_size=5, strides=1, padding='same', activation='elu')(
        conv_layer5)
    conv_layer7 = Conv1DBlock(filters=num_dense // 2, kernel_size=5, strides=1, padding='same', activation='elu')(
        conv_layer6)
    conv_layer8 = Conv1D(filters=1, kernel_size=3, strides=1, padding='same')(conv_layer7)
    a6 = Activation('sigmoid')(conv_layer8)

    model = tf.keras.models.Model(inputs=input_layer, outputs=a6)

    return model


########################################  QAE models  ##########################################


class QCNN_AE(tf.keras.models.Model):
    def __init__(self, snr_train=snr_train, use_onehot=False, circuit_idx=7, **kwargs):
        super(QCNN_AE, self).__init__(**kwargs)

        self.model_type = 'qcnn'
        self.noise_stddev = tf.sqrt(1 / (2 * R * (10 ** (snr_train / 10.0))))
        self.use_onehot = use_onehot

        ### Define layers
        last_dim = 2 if circuit_idx == 7 else 3
        weight_shapes = {"weights": (num_layers, 1, num_wires, last_dim)}
        self.qfilter1 = qml.qnn.KerasLayer(dict_circuit.get(f"{circuit_idx}{2}"), weight_shapes, output_dim=2,
                                           name='qfilter1')
        self.qfilter2 = qml.qnn.KerasLayer(dict_circuit.get(f"{circuit_idx}{2}"), weight_shapes, output_dim=2,
                                           name='qfilter2')
        self.qfilter3 = qml.qnn.KerasLayer(dict_circuit.get(f"{circuit_idx}{2}"), weight_shapes, output_dim=2,
                                           name='qfilter3')
        # self.qfilter4 = qml.qnn.KerasLayer(dict_circuit.get(f"{circuit_idx}{2}"), weight_shapes, output_dim=2, name='qfilter4')
        self.bn1 = BatchNormalization()
        self.elu1 = Activation('elu')
        self.reshape1 = Reshape((4, 1))

        self.norm = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * (K.l2_normalize(x, axis=1)),
                           name='norm_layer')
        # self.channel = Channel_Model(channel_type, self.noise_stddev)
        self.channel = Lambda(lambda x: Rayleigh_Channel_tf(x, self.noise_stddev), name='channel_layer')

        self.reshape2 = Reshape((16, 1))

        # Rx
        self.Rx = receiver(use_onehot=False)

    def calculate_padding(self, kernel_size, padding='valid'):
        if padding == 'same':
            total_pad = kernel_size - 1
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
        elif padding == 'valid':
            pad_left, pad_right = 0, 0
        else:
            raise ValueError("Padding type not supported. Please use 'same' or 'valid'.")

        return pad_left, pad_right

    def apply_padding(self, inputs, kernel_size, padding='valid'):
        pad_left, pad_right = self.calculate_padding(kernel_size, padding)
        if pad_left > 0 or pad_right > 0:
            padding = [[0, 0], [pad_left, pad_right], [0, 0]]
            inputs = tf.pad(inputs, paddings=padding, mode="CONSTANT")
        return inputs

    def qconv1d(self, inputs, qfilter, filters, kernel_size, strides, padding='valid'):
        inputs = self.apply_padding(inputs, padding)
        bs, input_dim, input_channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        output_dim = (input_dim - kernel_size) // strides + 1

        outputs = []
        for batch_idx in range(bs):
            batch_outputs = []
            for idx in range(0, output_dim * strides, strides):
                q_inputs = tf.squeeze(inputs[batch_idx, idx:idx + kernel_size, 0])
                q_inputs = tf.cast(q_inputs, tf.float32)
                q_outputs = qfilter(q_inputs)
                batch_outputs.append(q_outputs)

            batch_outputs = tf.stack(batch_outputs, axis=0)
            batch_outputs = tf.reshape(batch_outputs, (-1, filters))
            outputs.append(batch_outputs)

        x = tf.stack(outputs, axis=0)

        return x

    def qconv1d_batch(self, inputs, qfilter, filters, kernel_size, strides, padding='valid'):
        inputs = self.apply_padding(inputs, kernel_size, padding)
        bs, input_dim, inputs_channel = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        output_dim = (input_dim - kernel_size) // strides + 1

        outputs = []
        for k_idx in range(0, output_dim * strides, strides):
            for ch_idx in range(0, inputs_channel):
                q_inputs = tf.cast(inputs[:, k_idx:k_idx + kernel_size, ch_idx], tf.float32)
                q_outputs = tf.cast(qfilter(q_inputs), tf.float32)
                q_outputs = tf.reshape(q_outputs, (-1, filters))

                outputs.append(q_outputs)
        x = tf.stack(outputs, axis=1)

        return x

    def encode(self, inputs):
        # Rayleigh
        x1 = self.qconv1d_batch(inputs, self.qfilter1, filters=2, kernel_size=2, strides=2, padding='valid')
        x2 = self.qconv1d_batch(inputs, self.qfilter2, filters=2, kernel_size=2, strides=2, padding='valid')
        x = tf.concat([x1, x2], axis=1)
        # x = self.bn1(x)
        # x = self.elu1(x)
        x = self.qconv1d_batch(x, self.qfilter3, filters=2, kernel_size=2, strides=2, padding='valid')
        x_norm = self.norm(x)

        return x_norm

    def decode(self, y):
        # Rayeligh
        outputs = self.Rx(y)
        return outputs

    def call(self, inputs):
        x = self.encode(inputs)
        y_noisy = self.channel(x)
        outputs = self.decode(y_noisy)

        return outputs

    def test_bler_ber(self, test_datasize=int(1e6), save=False):
        if self.use_onehot:
            inputs = get_onehot_data(test_datasize, bit_num)
        else:
            inputs = get_binary_data(test_datasize, bit_num)
        SNR_range = list(np.linspace(0, 20, 41))
        ber_list, bler_list = [], []

        x = self.encode(inputs)
        print(f"Running BER BLER Test for QCNN {num_qubits}{bit_num} on {channel_type}!!!")
        for snr in SNR_range:
            SNR = 10 ** (snr / 10.0)
            noise_std = np.sqrt(1 / (2 * R * SNR))

            if channel_type == 'awgn':
                y_noisy = AWGN_Channel_tf(x, noise_std)
            else:
                y_noisy = Rayleigh_Channel_tf(x, noise_std)

            inputs_est = self.decode(y_noisy)
            if self.use_onehot:
                inputs_round = onehot_to_binary(inputs_est).numpy()
                inputs_binary = onehot_to_binary(inputs).numpy()
                ber_val = get_ber(inputs_binary, inputs_round)
                bler_val = np.sum(np.argmax(inputs_est, axis=1) != np.argmax(inputs, axis=1)) / test_datasize
            else:
                inputs_round = (np.rint(inputs_est)).astype(int)
                inputs_binary = inputs
                ber_val = get_ber(inputs_binary, inputs_round)
                bler_val = get_bler(inputs_binary, inputs_round)

            ber_list.append(ber_val)
            bler_list.append(bler_val)
            print('SNR: ', snr, 'BLER: ', bler_val, 'BER: ', ber_val)

            # free space
            gc.collect()
            K.clear_session()

            if ber_val == 0 and bler_val == 0:
                num_gap = int(len(SNR_range) - len(bler_list))
                bler_list += num_gap * [0.0]
                ber_list += num_gap * [0.0]
                break

        if save:
            file_path_bler = f"{num_qubits}{bit_num}_{self.model_type}/lists/bler_list_{self.model_type}_{num_qubits}{bit_num}_SNR{snr_train}.txt"
            file_path_ber = f"{num_qubits}{bit_num}_{self.model_type}/lists/ber_list_{self.model_type}_{num_qubits}{bit_num}_SNR{snr_train}.txt"

            np.savetxt(file_path_bler, bler_list)
            np.savetxt(file_path_ber, ber_list)

        return ber_list, bler_list

    def save_model_weights(self, filepath=None):
        filepath = f"{num_qubits}{bit_num}_{self.model_type}/models/model_ae_qcnn_{num_qubits}{bit_num}_SNR{snr_train}.keras" if filepath is None else filepath
        # filepath = os.path.join(os.getcwd(), filepath)
        print("filepath: ", filepath)
        self.save_weights(filepath)

        print(f"Successfully saved model for {self.model_type} {num_qubits}{bit_num}!!!")

    def load_model_weights(self, filepath=None):
        if not self.built:
            self.built = True
        filepath = f"{num_qubits}{bit_num}_{self.model_type}/models/model_ae_qcnn_{num_qubits}{bit_num}_SNR{snr_train}.keras" if filepath is None else filepath
        # filepath = os.path.join(os.getcwd(), filepath)
        print("filepath: ", filepath)
        self.load_weights(filepath)

        print(f"Successfully loaded model for {self.model_type} {num_qubits}{bit_num}!!!")


def train_qcnn(use_onehot=False, use_callback=False, retrain=False):
    create_folders(num_qubits, bit_num)
    model_qcnn = QCNN_AE(use_onehot=use_onehot)
    if retrain:
        model_qcnn.load_model_weights()

    lr = 0.001
    if use_onehot:
        x_train = y_train = get_onehot_data(50000, bit_num)
        x_val = y_val = get_onehot_data(100, bit_num)
        model_qcnn.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy',
                           metrics=['acc', ber_onehot])
    else:
        x_train = y_train = get_binary_data(50000, bit_num)
        x_val = y_val = get_binary_data(100, bit_num)
        model_qcnn.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                           metrics=['acc', ber_metric])

    checkpoint_path = f'{num_qubits}{bit_num}_qcnn/models/model_ae_qcnn_{num_qubits}{bit_num}_SNR{snr_train}.keras'
    # checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,  # Path to save the model file
        monitor='ber_metric',  # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode='min',  # Mode for the monitored metric ('min' for val_loss)
        save_weights_only=True,  # Save the whole model (not just weights)
        verbose=1  # Verbosity mode
    )
    if use_callback:
        lr_scheduler = CustomLearningRateScheduler()
        history = model_qcnn.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_val, y_val),
                                 callbacks=[lr_scheduler, checkpoint_callback])
    else:
        history = model_qcnn.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_val, y_val),
                                 callbacks=[checkpoint_callback])
    print(model_qcnn.summary())

    # model_qcnn.save_model_weights()
    save_history(history, 'qcnn', num_qubits, bit_num, snr_train, True)
    ber_list, bler_list = model_qcnn.test_bler_ber(save=True)

    return model_qcnn, history, ber_list, bler_list


set_seeds()
train_qcnn(use_onehot=False, use_callback=False, retrain=True)