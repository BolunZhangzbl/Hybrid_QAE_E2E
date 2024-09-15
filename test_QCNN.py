# -- Public Imports

# --Private Imports
from QCNN import *

# -- Global Variables

bit_num = 4
num_qubits = 4   # number of wires
R = float(bit_num / num_qubits)

snr_train = 7

# -- Functions


########################################  Test BLER  ##########################################

def test_qcnn(save=False):
    """"""
    model_qcnn = QCNN_AE(use_onehot=False)
    model_qcnn.load_model_weights()

    ber_list, bler_list = model_qcnn.test_bler_ber(save=save)

    return ber_list, bler_list


test_qcnn()