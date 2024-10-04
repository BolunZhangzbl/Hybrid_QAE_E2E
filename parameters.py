### parameters.py


# -- Public Imports
import pennylane as qml

# -- Global Variables

# -- Functions


class Config:
    def __init__(self):
        self.bit_num = 4
        self.num_qubits = 4
        self.use_onehot = True
        self.R = float(self.bit_num / self.num_qubits)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.channel_type = "rayleigh"

        self.snr_train = 7 if self.channel_type == 'awgn' else 10

        self.batch_size = 32
        self.num_layers = 3

        self.save = True

    def update(self, args):
        self.bit_num = args.bit_num
        self.num_qubits = args.num_qubits
        self.use_onehot = args.use_onehot
        self.channel_type = args.channel_type
        self.snr_train = 7 if self.channel_type == 'awgn' else 10
        self.save = args.save
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.R = float(self.bit_num / self.num_qubits)

        print("bit_num: ", self.bit_num)
        print("num_qubits: ", self.num_qubits)
        print("R: ", self.R)
        print("channel type:", self.channel_type)
        print("snr train: ", self.snr_train)


config = Config()
