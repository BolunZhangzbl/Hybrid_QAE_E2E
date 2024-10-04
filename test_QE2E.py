# -- Public Imports

# --Private Imports
from QE2E import *
from parameters import *

# -- Global Variables

# -- Functions


########################################  Test BLER  ##########################################

def test(config):
    """"""
    
    model_qae = QAE(config)
    model_qae.load_model_weights()
    
    test_size = int(1e6) if config.num_qubits==4 else int(1e5)
    
    # bler ber test
    ber_list, bler_list = model_qae.test_bler_ber(test_datasize=test_size, save=config.save)
   
