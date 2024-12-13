# -- Public Imports

# --Private Imports
from QE2E import *
from parameters import *

# -- Global Variables

# -- Functions


########################################  Test BLER  ##########################################

def test(config):
    """Function to test the QAE model for BLER and BER performance.

    Args:
        config: Configuration object with attributes such as num_qubits, bit_num, dev, snr_train, and save.

    This function evaluates the performance of a Quantum Autoencoder (QAE) model under different configurations
    and saves the results to text files if required.
    """
    # Initialize and load the QAE model
    model_qae = QAE(config)
    model_qae.load_model_weights()

    # Determine test data size based on the configuration
    if config.num_qubits == 4:
        test_size = int(1e6)
    else:
        test_size = int(1e5)

    # Adjust test size for non-default.qubit devices
    if config.dev.name != 'default.qubit':
        test_size = int(1e4)

        # Initialize lists to store intermediate BLER and BER results
        ber_list, bler_list = [], []

        # Perform multiple iterations for more robust results
        for _ in range(10):
            ber_tmp, bler_tmp = model_qae.test_bler_ber(test_datasize=test_size, save=False)
            ber_list.append(ber_tmp)
            bler_list.append(bler_tmp)

        # Compute mean BLER and BER over all iterations
        ber_list = np.mean(ber_list, axis=0)
        bler_list = np.mean(bler_list, axis=0)

        # Define file paths for saving results
        folder_path = f"{config.num_qubits}{config.bit_num}_onehot/lists/"
        file_path_bler = f"{folder_path}bler_list_qae_{config.num_qubits}{config.bit_num}_SNR{config.snr_train}.txt"
        file_path_ber = f"{folder_path}ber_list_qae_{config.num_qubits}{config.bit_num}_SNR{config.snr_train}.txt"

        # Save BLER and BER results to files
        np.savetxt(file_path_bler, bler_list)
        np.savetxt(file_path_ber, ber_list)

    else:
        # Direct testing for default.qubit devices
        ber_list, bler_list = model_qae.test_bler_ber(test_datasize=test_size, save=config.save)

    return ber_list, bler_list