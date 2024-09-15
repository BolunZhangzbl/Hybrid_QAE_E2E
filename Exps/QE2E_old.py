# -- Public Imports
import gc
import os
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Conv1D, Activation, BatchNormalization, Reshape, Flatten


# --Private Imports

# -- Global Variables

bit_num = 4
num_qubits = 7   # number of wires
R = float(bit_num / num_qubits)

batch_size = 32
snr_train = 7

num_layers = 12
weights_init = 0.01 * np.random.rand(num_layers, 1, num_qubits, 3, requires_grad=True)
inputs_weights_init = 0.01 * np.random.rand(num_layers, num_qubits, requires_grad=True)

dev = qml.device("default.qubit", wires=num_qubits)

tx_id, rx_id = 'wDR_new', 'wDR_new'


# -- Functions

########################################  Utils function for layers  ########################################


def weighted_reuploading_layer(inputs, layer_weights):
    weighted_inputs = inputs * layer_weights[0, :, 0]   # (32,num_qubits) * (num_qubits, )

    qml.templates.AngleEmbedding(weighted_inputs, wires=range(num_qubits), rotation='Z')
    StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))


def weighted_reuploading_layer_new(inputs, layer_weights, layer_inputs_weights):
    weighted_inputs = inputs * layer_inputs_weights   # (32,num_qubits) * (num_qubits, )

    qml.templates.AngleEmbedding(weighted_inputs, wires=range(num_qubits), rotation='Z')
    StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))


########################################  Utils function for circuits  ########################################


@qml.qnode(dev, interface='tf', diff_method='best')
def circuit1(inputs, weights):
    """
    Normal circuit1 without Data re-uploading
    :param inputs:
    :param weights:
    :return:
    """
    qml.templates.AngleEmbedding(inputs * np.pi, wires=range(num_qubits), rotation='Z')

    for layer_weights in weights:
        StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


@qml.qnode(dev, interface='tf', diff_method='best')
def circuit2(inputs, weights):
    """
    Normal circuit2 without Data re-uploading
    :param inputs:
    :param weights:
    :return:
    """
    qml.templates.AngleEmbedding(inputs * np.pi, wires=range(num_qubits), rotation='Z')
    StronglyEntanglingLayers(tf.squeeze(weights), wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


@qml.qnode(dev, interface='tf', diff_method='best')
def circuit_DR(inputs, weights):
    """
    Circuit with Data re-uploading
    :param inputs:
    :param weights:
    :return:
    """
    for layer_weights in weights:
        qml.templates.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Z')
        StronglyEntanglingLayers(layer_weights, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


@qml.qnode(dev, interface='tf', diff_method='best')
def circuit_wDR(inputs, weights):
    """
    Circuit with weighted Data re-uploading
    :param inputs:
    :param weights:
    :return:
    """
    for layer_weights in weights:
        weighted_reuploading_layer(inputs, layer_weights)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


@qml.qnode(dev, interface='tf', diff_method='best')
def circuit_wDR_new(inputs, weights, inputs_weights):
    """
    Circuit with weighted Data re-uploading
    :param inputs:
    :param weights:
    :return:
    """
    for layer_weights, layer_inputs_weights in zip(weights, inputs_weights):
        weighted_reuploading_layer_new(inputs, layer_weights, layer_inputs_weights)

    return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]


dict_circuit = {'1': circuit1, '2': circuit2, 'DR': circuit_DR, 'wDR': circuit_wDR, 'wDR_new': circuit_wDR_new}



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


def AWGN_Channel_verify(x):
    results = Add()([x, tf.ones_like(x)])
    return results


########################################  Get Tx & Rx models  ##########################################



def get_tx_rx(model):
    """"""
    assert isinstance(model, tf.keras.models.Model)

    # Get channel layer idx
    for layer_idx, layer in enumerate(model.layers):
        if layer.name == 'channel_layer':
            break

    # Extract encoder and decoder layers from the model
    encoder_layers = model.layers[:layer_idx]
    decoder_layers = model.layers[layer_idx+1:]

    # Create encoder model
    encoder_input = model.input
    encoder_output = encoder_layers[-1].output
    encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

    # Create decoder model
    decoder_input = tf.keras.Input((num_qubits))
    decoder_output = decoder_input
    for layer in decoder_layers:
        decoder_output = layer(decoder_output)
    decoder_model = tf.keras.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

    return encoder_model, decoder_model


def get_qae(use_onehot=False, num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train, use_verify=False):
    """"""
    global R
    R = float(bit_num / num_qubits)

    print(f"Tx: {tx_id} - Rx: {rx_id}")
    tx_circuit = dict_circuit.get(str(tx_id))
    rx_circuit = dict_circuit.get(str(rx_id))
    print(tx_circuit == circuit2)
    print(rx_circuit == circuit_wDR)

    noise_stddev = np.sqrt(1 / (2 * R * (10 ** (snr_train / 10.0))))
    weight_shapes_tx = {'weights': (num_layers, 1, num_qubits, 3), 'inputs_weights': (num_layers, num_qubits)}
    weight_shapes_rx = {'weights': (num_layers, 1, num_qubits, 3), 'inputs_weights': (num_layers, num_qubits)}
    if use_onehot:
        num_dense = 2**bit_num
        activation = 'softmax'
    else:
        num_dense = bit_num
        activation = 'sigmoid'

    input_layer = Input((num_dense, ))
    dense_layer1 = Dense(num_qubits, activation="linear", name='dense_layer1')(input_layer)
    encoder_layer = qml.qnn.KerasLayer(tx_circuit, weight_shapes_tx, output_dim=num_qubits, name='encoder_layer')(dense_layer1)
    # norm_layer = Lambda(lambda x: x / tf.sqrt(tf.reduce_mean(x**2)), name='norm_layer')(encoder_layer)
    norm_layer = Lambda(lambda x: tf.sqrt(tf.cast(num_qubits, tf.float32)) * (K.l2_normalize(x, axis=1)), name='norm_layer')(encoder_layer)
    if use_verify:
        channel_layer = Lambda(lambda x: AWGN_Channel_verify(x), name='channel_layer')(norm_layer)
    else:
        channel_layer = Lambda(lambda x: AWGN_Channel_tf(x, noise_stddev), name='channel_layer')(norm_layer)
    decoder_layer = qml.qnn.KerasLayer(rx_circuit, weight_shapes_rx, output_dim=num_qubits, name='decoder_layer')(channel_layer)
    dense_layer2 = Dense(num_dense, activation=activation, name='dense_layer2')(decoder_layer)

    model_qae = tf.keras.Model(inputs=input_layer, outputs=dense_layer2)

    return model_qae


########################################  Save Models  ##########################################

def save_history(history, ae_type='qae', num_qubits=num_qubits, bit_num=bit_num, snr_train=7, save=False):
    assert ae_type in ('qae', 'ae')

    history_dict = history.history if hasattr(history, 'history') else history
    if save:
        filename = f"history_{ae_type}_{num_qubits}{bit_num}_SNR{snr_train}.npz"
        file_path = os.path.join(os.getcwd(), f'{num_qubits}{bit_num}_binary/history/', filename)
        np.savez(file_path, **history_dict)

    print(f"Successfully saved history dict for {ae_type} {num_qubits}{bit_num}!!!")


def save_ae(model_ae, model_type='qae', num_qubits=num_qubits, bit_num=bit_num,
            snr_train=7, save=False):
    assert isinstance(model_ae, tf.keras.models.Model)
    assert model_type in ('qae', 'ae')

    if save:
        file_path_ae = os.path.join(
            os.getcwd(),
            f"{num_qubits}{bit_num}_binary/models/model_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras")
        model_ae.save_weights(file_path_ae)

    print(f"Successfully saved Autoencoder model for {model_type} {num_qubits}{bit_num}!!!")


def save_tx_rx(model_tx, model_rx, model_type='qae', num_qubits=num_qubits,
               bit_num=bit_num, snr_train=7, save=False):
    assert isinstance(model_tx, tf.keras.models.Model)
    assert isinstance(model_rx, tf.keras.models.Model)
    assert model_type in ('qae', 'ae')

    if save:
        file_path_tx = os.path.join(
            os.getcwd(),
            f"{num_qubits}{bit_num}_binary/models/model_tx_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras")
        file_path_rx = os.path.join(
            os.getcwd(),
            f"{num_qubits}{bit_num}_binary/models/model_rx_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras")
        model_tx.save_weights(file_path_tx)
        model_rx.save_weights(file_path_rx)

    print(f"Successfully saved the tx rx models for {model_type} {num_qubits}{bit_num}!!!")


########################################  Load Models  ##########################################

def load_ae(model_ae, model_type='qae', num_qubits=num_qubits, bit_num=bit_num, snr_train=7):
    """"""
    model_ae_path = f"{num_qubits}{bit_num}_binary/models/model_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras"
    model_ae_path = os.path.join(os.getcwd(), model_ae_path)

    model_ae.load_weights(model_ae_path)

    print(f"Successfully loaded Autoencoder model for {model_type} {num_qubits}{bit_num}!!!")

    return model_ae


def load_tx_rx(model_tx, model_rx, model_type='qae', num_qubits=num_qubits, bit_num=bit_num, snr_train=7):
    """"""
    try:
        model_tx_path = os.path.join(
            os.getcwd(),
            f"{num_qubits}{bit_num}_binary/models/model_tx_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras")
        model_rx_path = os.path.join(
            os.getcwd(),
            f"{num_qubits}{bit_num}_binary/models/model_rx_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.keras")

        model_tx.load_weights(model_tx_path)
        model_rx.load_weights(model_rx_path)

        print(f"Successfully loaded the tx rx models for {model_type} {num_qubits}{bit_num}!!!")
        return model_tx, model_rx
    except Exception as e:
        print(f"tx rx models not found for {model_type} {num_qubits}{bit_num}!!!")
        return


def get_pretrained_tx_rx_wrapper(model_type='qae', use_onehot=False,
                                 num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train):
    """"""
    assert model_type in ('qae', 'ae')

    model = get_qae(use_onehot=use_onehot, num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train)
    model_tx, model_rx = get_tx_rx(model)
    model_tx, model_rx = load_tx_rx(model_tx, model_rx, model_type, num_qubits, bit_num, snr_train=snr_train)

    return model_tx, model_rx


########################################  Utils Func to save & read  ##########################################

def create_folders(num_qubits=num_qubits, bit_num=bit_num):
    dirs = [f"{num_qubits}{bit_num}_binary/history",
            f"{num_qubits}{bit_num}_binary/models",
            f"{num_qubits}{bit_num}_binary/lists"]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory: {dir}!")
    return


########################################  Verification of savings  ##########################################

def verify_tx_rx(model_qae, model_tx_ver, model_rx_ver, bit_num=bit_num):
    """"""
    print(model_qae.summary())
    print(model_tx_ver.summary())
    print(model_rx_ver.summary())
    # verify tx models
    data = tf.random.normal((32, bit_num))
    x = model_tx_ver(data)

    # verify rx models
    y = AWGN_Channel_verify(x)
    data_est1 = model_qae(data)
    data_est2 = model_rx_ver(y)

    print("data_est1: ", data_est1[:2])
    print("data_est2: ", data_est2[:2])

    if tf.reduce_all(tf.equal(data_est1, data_est2)):
        print("Verified the saved models Successfully")
    else:
        raise NotImplementedError("Verification of saved models failed!")


########################################  Call BLER  ##########################################


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
    return tf.reduce_mean(tf.cast(tf.not_equal(tf.math.round(y_pred), y_true), tf.float32))


def ber_onehot(y_true, y_pred):

    y_true_binary = onehot_to_binary(y_true)
    y_pred_binary = onehot_to_binary(y_pred)

    return tf.reduce_mean(tf.cast(tf.not_equal(y_pred_binary, y_true_binary), tf.float32))


def get_binary_data(data_size, bit_num=bit_num):
    """"""
    data = np.random.binomial(1, 0.5, size=(data_size, bit_num))
    return data


def get_onehot_data(data_size, bit_num=bit_num):
    # Generate random integer indices between 0 and num_bits-1
    num_bits_onehot = 2**bit_num
    random_indices = np.random.randint(0, num_bits_onehot, size=data_size)

    # Create an array of zeros with shape (data_size, num_bits)
    onehot_data = np.zeros((data_size, num_bits_onehot), dtype=int)

    # Set the appropriate indices to 1 for one-hot encoding
    onehot_data[np.arange(data_size), random_indices] = 1

    return onehot_data


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


def call_ber_bler(model_tx, model_rx, model_type='qae', channel_type='awgn', test_datasize=int(1e5),
                  SNR_low=0, SNR_high=20, SNR_num=41, snr_train=snr_train,
                  num_qubits=num_qubits, bit_num=bit_num, use_onehot=False, save=False):
    """"""
    print(model_tx.summary())
    print(model_rx.summary())
    global R
    R = float(bit_num / num_qubits)

    assert channel_type in ('awgn', 'rician', 'rayleigh')
    if use_onehot:
        inputs = get_onehot_data(test_datasize, bit_num)
    else:
        inputs = get_binary_data(test_datasize, bit_num)

    SNR_range = list(np.linspace(SNR_low, SNR_high, SNR_num))
    ber_list, bler_list = [], []

    x = model_tx(inputs)
    print(f"Running BER BLER Test for {model_type} {num_qubits}{bit_num}!!!")
    for snr in range(0, len(SNR_range)):
        SNR = 10 ** (SNR_range[snr] / 10.0)
        noise_std = np.sqrt(1 / (2 * R * SNR))

        y_noisy = AWGN_Channel_tf(x, noise_std)
        inputs_est = model_rx(y_noisy)

        if use_onehot:
            inputs_round = onehot_to_binary(inputs_est).numpy()
            inputs_binary = onehot_to_binary(inputs).numpy()
        else:
            inputs_round = (np.rint(inputs_est)).astype(int)
            inputs_binary = inputs

        ber_val = get_ber(inputs_binary, inputs_round)
        ber_list.append(ber_val)
        bler_val = get_bler(inputs_binary, inputs_round)
        bler_list.append(bler_val)
        print('SNR: ', SNR_range[snr], 'BLER: ', bler_val, 'BER: ', ber_val)

        # free space
        gc.collect()
        K.clear_session()

        if ber_val == 0 and bler_val == 0:
            num_gap = int(len(SNR_range) - len(bler_list))
            bler_list += num_gap * [0.0]
            ber_list += num_gap * [0.0]
            break

    if save:
        file_path_bler = f"{num_qubits}{bit_num}_binary/lists/bler_list_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.txt"
        file_path_ber = f"{num_qubits}{bit_num}_binary/lists/ber_list_{model_type}_{num_qubits}{bit_num}_SNR{snr_train}.txt"

        np.savetxt(file_path_bler, bler_list)
        np.savetxt(file_path_ber, ber_list)

    return ber_list, bler_list


# Main Function
def verify(num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train, use_onehot=False):
    """
    Function to verify the E2E system
    :param num_qubits:
    :param bit_num:
    :param snr_train:
    :param use_onehot:
    :return:
    """
    model_qae_ver = get_qae(use_onehot=use_onehot, num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train,
                            use_verify=True)
    if use_onehot:
        x_train = y_train = get_onehot_data(50, bit_num)
        x_val = y_val = get_onehot_data(50, bit_num)
        model_qae_ver.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                              metrics=['acc', ber_onehot])
    else:
        x_train = y_train = get_binary_data(50, bit_num)
        x_val = y_val = get_binary_data(50, bit_num)
        model_qae_ver.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                              metrics=['acc', ber_metric])
    model_qae_ver.fit(x_train, y_train, epochs=1, batch_size=50, validation_data=(x_val, y_val))
    model_tx_ver, model_rx_ver = get_tx_rx(model_qae_ver)
    verify_tx_rx(model_qae_ver, model_tx_ver, model_rx_ver, bit_num=bit_num)


def train(num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train,
          use_onehot=False, use_callback=False, use_verify=False, save=False):
    """"""
    if use_verify:
        verify(num_qubits, bit_num, snr_train, use_onehot)

        return
    else:
        create_folders(num_qubits=num_qubits, bit_num=bit_num)
        model_qae = get_qae(use_onehot=use_onehot, num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train)
        if use_onehot:
            x_train = y_train = get_onehot_data(5000, bit_num)
            x_val = y_val = get_onehot_data(500, bit_num)
            model_qae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                              metrics=['acc', ber_onehot])
        else:
            x_train = y_train = get_binary_data(5000, bit_num)
            x_val = y_val = get_binary_data(500, bit_num)
            model_qae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                              metrics=['acc', ber_metric])
        print(model_qae.summary())

        if use_callback:
            lr_scheduler = CustomLearningRateScheduler()
            checkpoint_path = os.path.join(os.getcwd(), f'model_qae_{num_qubits}{bit_num}_SNR{snr_train}.keras')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                # Path to save the model file
                monitor='ber_metric',  # Metric to monitor
                save_best_only=True,  # Save only the best model
                mode='min',  # Mode for the monitored metric ('min' for val_loss)
                save_weights_only=True,  # Save the whole model (not just weights)
                verbose=1  # Verbosity mode
            )
            history_qae = model_qae.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val),
                                        callbacks=[lr_scheduler, checkpoint_callback])
            model_qae = model_qae.load_weights(checkpoint_path)
        else:
            history_qae = model_qae.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

        # save models
        model_tx, model_rx = get_tx_rx(model_qae)
        save_ae(model_qae, 'qae', num_qubits, bit_num, snr_train, save=save)
        save_tx_rx(model_tx, model_rx, 'qae', num_qubits, bit_num, snr_train, save=save)
        save_history(history_qae, 'qae', num_qubits, bit_num, snr_train, save=save)

        # Run & save bler, ber
        ber_list, bler_list = call_ber_bler(model_tx, model_rx, model_type='qae', num_qubits=num_qubits, bit_num=bit_num,
                                            use_onehot=use_onehot, snr_train=snr_train, save=save)
        return model_qae, history_qae, ber_list, bler_list