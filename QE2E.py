# -- Public Imports
import gc
import os
import random
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Activation, Reshape, Flatten


# --Private Imports


# -- Global Variables


# -- Functions

def create_circuit(dev, num_qubits, bit_num, complex=True):
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

    def layer2(layer_weights):
        """
        q layer for AWGN
        :param layer_weights:
        :return:
        """
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
        for wire in range(num_qubits):
            qml.Rot(*layer_weights[0, wire] * np.pi, wires=wire)
        for wire in range(0, num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1)])

    @qml.qnode(dev, interface='tf', diff_method='best')
    def qcircuit_complex(inputs, weights):

        qml.templates.AmplitudeEmbedding(inputs, wires=range(bit_num), normalize=True)

        for layer_weights in weights:
            layer1(layer_weights)

        return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]

    @qml.qnode(dev, interface='tf', diff_method='best')
    def qcircuit_real(inputs, weights):

        qml.templates.AmplitudeEmbedding(inputs, wires=range(bit_num), normalize=True)

        for layer_weights in weights:
            layer2(layer_weights)

        return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]

    if complex:
        return qcircuit_complex
    else:
        return qcircuit_real


########################################  QAE, Channel  ########################################


def ls_equal(y, h):
    """"""
    y_complex = tf.complex(y[:, :, 0], y[:, :, 1])
    h_complex = tf.complex(h[:, :, 0], h[:, :, 1])
    x_equal = y_complex / h_complex

    return tf.stack([tf.math.real(x_equal), tf.math.imag(x_equal)], axis=-1)


def ls_est(xp, yp):
    """"""
    y_complex = tf.complex(yp[:, :, 0], yp[:, :, 1])
    x_complex = tf.complex(xp[:, :, 0], xp[:, :, 1])
    h_est = y_complex / x_complex

    return tf.stack([tf.math.real(h_est), tf.math.imag(h_est)], axis=-1)


def convolve(xr, xi, hr, hi, nr, ni):
    yr = hr*xr - hi*xi + nr
    yi = hr*xi + hi*xr + ni

    return tf.stack([yr, yi], axis=-1)


def Rayleigh_Channel_tf(x, noise_stddev):
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]

    H_R = tf.random.normal(K.shape(x_real[:, :1]), mean=0, stddev=tf.sqrt(1/2))
    H_I = tf.random.normal(K.shape(x_imag[:, :1]), mean=0, stddev=tf.sqrt(1/2))

    noise_r = tf.random.normal(K.shape(x_real), mean=0, stddev=noise_stddev)
    noise_i = tf.random.normal(K.shape(x_imag), mean=0, stddev=noise_stddev)

    # Concatenating the real and imaginary components of received signal
    y = convolve(x_real, x_imag, H_R, H_I, noise_r, noise_i)

    # Get the perfect channel response
    h_est = K.stack([H_R, H_I], axis=-1)

    # Concatenate the impaired signal and the estimated channel response
    results = ls_equal(y, h_est)

    return results
    

def AWGN_Channel_tf(x, noise_stddev):
    noise = tf.random.normal(K.shape(x), mean=0, stddev=noise_stddev)
    results = Add()([x, noise])

    return results


########################################  QAE Model ##########################################
    

def receiver_complex(num_qubits, bit_num):
    """"""
    ipl = Input((num_qubits, 2*1))
    fl1 = Flatten()(ipl)
    d3 = Dense(2**bit_num, activation='relu')(fl1)
    d4 = Dense(2**bit_num, activation='softmax')(d3)
    
    model = tf.keras.models.Model(inputs=ipl, outputs=d4)
    
    return model
    
    
def receiver_real(num_qubits, bit_num):
    """"""
    ipl = Input((num_qubits, ))
    d3 = Dense(2**bit_num, activation='relu')(ipl)
    d4 = Dense(2**bit_num, activation='softmax')(d3)
    
    model = tf.keras.models.Model(inputs=ipl, outputs=d4)
    
    return model
    
    
class QAE(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(QAE, self).__init__(**kwargs)

        set_global_var(config)
        
        self.model_type = 'qae'
        self.noise_stddev = tf.sqrt(1 / (2 * R * (10 ** (snr_train / 10.0))))
        self.use_onehot = config.use_onehot
        self.num_qubits = config.num_qubits
        self.bit_num = config.bit_num
        self.channel_type = config.channel_type
        self.snr_train = config.snr_train
    
        if self.channel_type != 'awgn':
            qcircuit_complex = create_circuit(config.dev, self.num_qubits, self.bit_num, complex=True)

            # Tx
            weight_shapes = {"weights": (num_layers, 1, self.num_qubits, 1)}
            self.q1 = qml.qnn.KerasLayer(qcircuit_complex, weight_shapes, output_dim=self.num_qubits, name='q1')
            self.q2 = qml.qnn.KerasLayer(qcircuit_complex, weight_shapes, output_dim=self.num_qubits, name='q2')
            self.norm = Lambda(lambda x: tf.sqrt(tf.cast(self.num_qubits, tf.float32)) * (K.l2_normalize(x, axis=[1, 2])), name='norm_layer')
            
            # Rx
            self.Rx = receiver_complex(self.num_qubits, self.bit_num)
            
            # Channel
            self.channel =  Lambda(lambda x: Rayleigh_Channel_tf(x, self.noise_stddev), name='channel_layer')
        else:
            qcircuit_real = create_circuit(config.dev, self.num_qubits, self.bit_num, complex=False)

            # Tx
            weight_shapes = {"weights": (5, 1, self.num_qubits, 3)}
            self.q1 = qml.qnn.KerasLayer(qcircuit_real, weight_shapes, output_dim=self.num_qubits, name='q1')
            self.norm = Lambda(lambda x: tf.sqrt(tf.cast(self.num_qubits, tf.float32)) * (K.l2_normalize(x, axis=[1])), name='norm_layer')
            
            # Rx
            self.Rx = receiver_real(self.num_qubits, self.bit_num)
        
            # Channel
            self.channel = Lambda(lambda x: AWGN_Channel_tf(x, self.noise_stddev), name='channel_layer')

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

    def apply_padding(self, inputs, kernel_size, padding='valid', circular=False):
        pad_left, pad_right = self.calculate_padding(kernel_size, padding)
        if pad_left>0 or pad_right>0:
            if circular:
                if pad_left>0:
                    inputs = tf.concat([inputs[:, -pad_left:], inputs], axis=1)
                if pad_right>0:
                    inputs = tf.concat([inputs, inputs[:, :pad_right]], axis=1)
            else:
                padding = [[0, 0], [pad_left, pad_right], [0, 0]]
                inputs = tf.pad(inputs, paddings=padding, mode="CONSTANT")
        return inputs
    
    def qconv1d_batch(self, inputs, qfilter, filters, kernel_size, strides, padding='valid', circular=False):
        inputs = self.apply_padding(inputs, kernel_size, padding, circular=circular)
        bs, input_dim = tf.shape(inputs)[0], tf.shape(inputs)[1]
        output_dim = (input_dim - kernel_size) // strides + 1

        outputs = []
        for k_idx in range(0, output_dim*strides, strides):
            q_inputs = tf.cast(inputs[:, k_idx:k_idx + kernel_size], tf.float32)
            q_outputs = tf.cast(qfilter(q_inputs), tf.float32)
            q_outputs = tf.reshape(q_outputs, (-1, filters))

            outputs.append(q_outputs)
        x = tf.stack(outputs, axis=1)

        return x

    def encode(self, inputs):
        if channel_type != 'awgn':
            # Rayleigh
            x1 = self.q1(inputs)
            x2 = self.q2(inputs)
            
            x = tf.stack([x1, x2], axis=-1)
        else:
            # AWGN
            x = self.q1(inputs)

        x = self.norm(x)
        
        return x
        
    def decode(self, y):
        # Rayleigh
        outputs = self.Rx(y)
         
        return outputs
        
    def call(self, inputs):
        x = self.encode(inputs)
        y_noisy = self.channel(x)
        outputs = self.decode(y_noisy)
        
        return outputs

    def test_bler_ber(self, test_datasize=int(1e6), save=False):

        if self.use_onehot:
            inputs = get_onehot_data(test_datasize, self.bit_num)
        else:
            inputs = get_binary_data(test_datasize, self.bit_num)
        SNR_range = list(np.linspace(0, 20, 41))
        ber_list, bler_list = [], []

        x = self.encode(inputs)
        print(f"Running BER BLER Test for QAE {self.num_qubits}{self.bit_num} on {self.channel_type}!!!")
        for snr in SNR_range:
            SNR = 10 ** (snr / 10.0)
            noise_std = np.sqrt(1 / (2 * R * SNR))
            
            if self.channel_type == 'rayleigh':
                y_noisy = Rayleigh_Channel_tf(x, noise_std)
            else:
                y_noisy = AWGN_Channel_tf(x, noise_std)

            inputs_est = self.decode(y_noisy)
            if self.use_onehot:
                inputs_round = onehot_to_binary(inputs_est).numpy()
                inputs_binary = onehot_to_binary(inputs).numpy()
                ber_val = get_ber(inputs_binary, inputs_round)
                # bler_val = get_bler(inputs_binary, inputs_round)
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
            file_path_bler = f"{self.num_qubits}{self.bit_num}_onehot/lists/bler_list_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.txt"
            file_path_ber = f"{self.num_qubits}{self.bit_num}_onehot/lists/ber_list_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.txt"

            np.savetxt(file_path_bler, bler_list)
            np.savetxt(file_path_ber, ber_list)

        return ber_list, bler_list
        
        
    def save_model_weights(self, filepath=None):
        filepath = f"{num_qubits}{bit_num}_onehot/models/model_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.keras" if filepath is None else filepath
        print("filepath: ", filepath)
        self.save_weights(filepath)
        
        print(f"Successfully saved model for {self.model_type} onehot {self.num_qubits}{self.bit_num}!!!")


    def load_model_weights(self, filepath=None):
        if not self.built:
            self.built = True
        filepath = f"{num_qubits}{bit_num}_onehot/models/model_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.keras" if filepath is None else filepath
        print("filepath: ", filepath)
        self.load_weights(filepath)
        
        print(f"Successfully loaded model for {self.model_type} onehot {self.num_qubits}{self.bit_num}!!!")
        
        
    def test_pilot_power(self, snr=10, test_datasize=int(1e6), save=False):
        global x_pilots
        
        SNR = 10 ** (snr / 10.0)
        pilot_noise_std = np.sqrt(1 / (2 * R * SNR))
        
        pilot_power_range = list(np.linspace(1, 20, 20))
        pilot_ber_list, pilot_bler_list = [], []
        
        inputs = get_onehot_data(test_datasize, self.bit_num)
        
        print(f"Running Pilot Power Test for QAE {self.num_qubits}{self.bit_num} on {self.channel_type}!!!")
        x = self.encode(inputs)
        for power in pilot_power_range:
            x_pilots = tf.random.normal((test_datasize, 1, 2))
            x_pilots = tf.sqrt(tf.cast(power, tf.float32)) * K.l2_normalize(x_pilots, axis=[1,2])
            
            y_noisy = Rayleigh_Channel_tf(x, pilot_noise_std)
            
            inputs_est = self.decode(y_noisy)
            
            inputs_round = onehot_to_binary(inputs_est).numpy()
            inputs_binary = onehot_to_binary(inputs).numpy()
            ber_val = get_ber(inputs_binary, inputs_round)
            # bler_val = get_bler(inputs_binary, inputs_round)
            bler_val = np.sum(np.argmax(inputs_est, axis=1) != np.argmax(inputs, axis=1)) / test_datasize
            
            pilot_ber_list.append(ber_val)
            pilot_bler_list.append(bler_val)
            print('SNR: ', snr, 'pilot power: ', power, 'BLER: ', bler_val, 'BER: ', ber_val)
        
        # save
        if save:
            file_path_bler = f"{self.num_qubits}{self.bit_num}_onehot/lists/pilot_bler_list_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.txt"
            file_path_ber = f"{self.num_qubits}{self.bit_num}_onehot/lists/pilot_ber_list_{self.model_type}_{self.num_qubits}{self.bit_num}_SNR{self.snr_train}.txt"

            np.savetxt(file_path_bler, pilot_bler_list)
            np.savetxt(file_path_ber, pilot_ber_list)
        
        return pilot_ber_list, pilot_bler_list
        
        

########################################  Save History  ##########################################


def save_history(history, num_qubits, bit_num, snr_train=10, save=False):
    ae_type = 'qae'
    history_dict = history.history if hasattr(history, 'history') else history
    if save:
        filename = f"history_{ae_type}_{num_qubits}{bit_num}_SNR{snr_train}.npz"
        filepath = os.path.join(f'{num_qubits}{bit_num}_onehot/history/', filename)
        np.savez(filepath, **history_dict)

    print(f"Successfully saved history dict for {ae_type} {num_qubits}{bit_num}!!!")


########################################  Utils Func to create folders  ##########################################


def create_folders(num_qubits, bit_num):
    dirs = [f"{num_qubits}{bit_num}_onehot/history",
            f"{num_qubits}{bit_num}_onehot/models",
            f"{num_qubits}{bit_num}_onehot/lists"]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory: {dir}!")
    return
    

########################################  Set Seeds  ##########################################

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
def set_global_var(config):
    """"""
    global num_qubits, bit_num, use_onehot, dev, snr_train, R, channel_type, batch_size, num_layers, save

    num_qubits = config.num_qubits
    bit_num = config.bit_num
    use_onehot = config.use_onehot
    dev = qml.device("default.qubit", wires=num_qubits)
    channel_type = config.channel_type
    batch_size = 32
    num_layers = 3
    snr_train = 7 if channel_type == 'awgn' else 10
    save = config.save

    R = float(bit_num / num_qubits)
    print("bit_num: ", bit_num)
    print("num_qubits: ", num_qubits)
    print("R: ", R)
    print("channel type:", channel_type)
    print("snr train: ", snr_train)



########################################  Utils Func  ##########################################


def onehot_to_binary(onehot_tensor):
    # Convert onehot tensor to integer tensor
    onehot_tensor = tf.squeeze(onehot_tensor)
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


def bler_onehot(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    y_true, y_pred = tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)
    errors = tf.not_equal(y_pred, y_true)
    bler = tf.reduce_mean(tf.cast(errors, tf.float32))
    
    return bler
    
    
def bler_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    errors = tf.not_equal(y_true, y_pred)
    block_errors = tf.reduce_any(errors, axis=1)
    bler = tf.reduce_mean(tf.cast(block_errors, tf.float32))
    
    return bler
    

def get_binary_data(data_size, bit_num):
    """"""
    data = np.random.binomial(1, 0.5, size=(data_size, bit_num))
    return data


def get_onehot_data(data_size, bit_num):
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
    

def hybrid_loss(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    
    # cc loss
    cc_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    cc_loss = tf.reduce_mean(tf.cast(cc_loss_obj(y_true, y_pred), tf.float32))
    
    # bler loss
    y_true, y_pred = tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)
    errors = tf.not_equal(y_pred, y_true)
    bler = tf.reduce_mean(tf.cast(errors, tf.float32))
    
    # hybrid loss
    loss = 0.999*bler + 0.001*cc_loss
    
    return loss


def train(config):
    """"""

    lr = 0.001
    create_folders(num_qubits=config.num_qubits, bit_num=config.bit_num)
    set_seeds(2)
    model_qae = QAE(config)
    if config.retrain:
        model_qae.load_model_weights()
        
    if use_onehot:
        x_train = y_train = get_onehot_data(50016, config.bit_num)
        x_val = y_val = get_onehot_data(128, config.bit_num)
        model_qae.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy',
                          metrics=['acc', ber_onehot, bler_onehot])
    else:
        x_train = y_train = get_binary_data(50016, config.bit_num)
        x_val = y_val = get_binary_data(128, config.bit_num)
        model_qae.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                          metrics=['acc', ber_metric])
    
    checkpoint_path = f"{config.num_qubits}{config.bit_num}_onehot/models/model_qae_{config.num_qubits}{config.bit_num}_SNR{config.snr_train}.keras"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='bler_onehot',
        save_best_only=True, 
        mode='min',
        save_weights_only=True,
        verbose=1,
    )
    history_qae = model_qae.fit(x_train, y_train, epochs=276, batch_size=32, validation_data=(x_val, y_val),
                                callbacks=[checkpoint_callback])
    print(model_qae.summary())

    # save models
    model_qae.load_model_weights()
    save_history(history_qae, config.num_qubits, config.bit_num, config.snr_train, save=True)

    # Run & save bler, ber
    set_seeds(3)
    ber_list, bler_list = model_qae.test_bler_ber(save=True)
    
    return model_qae, history_qae, ber_list, bler_list