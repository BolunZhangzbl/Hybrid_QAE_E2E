# -- Public Imports

# --Private Imports
from QE2E import *

# -- Global Variables

bit_num = 4
num_qubits = 4   # number of wires
R = float(bit_num / num_qubits)

channel_type = 'rayleigh'

snr_train = 7 if channel_type == 'awgn' else 10

# -- Functions


########################################  Test BLER  ##########################################

def test(num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train, use_onehot=True, save=True):
    """"""
    
    model_qae = QAE(use_onehot=use_onehot)
    model_qae.load_model_weights()
    
    test_size = int(1e6)
    
    # bler ber test
    ber_list, bler_list = model_qae.test_bler_ber(test_datasize=test_size, save=save)
   
