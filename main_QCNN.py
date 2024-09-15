# -- Public Imports
import argparse

# -- Private Imports
import test_QCNN as tq
import QCNN as q

# -- Functions


def main_qcnn(train_mode=True, num_qubits=4, bit_num=4, snr_train=10, use_onehot=False, save=True):
    """
    Main function to either train or test the model based on the provided arguments.
    """
    q.set_global_var(num_qubits_val=num_qubits, bit_num_val=bit_num, snr_train_val=snr_train,
                     use_onehot_val=use_onehot, num_epochs_val=180)
    if train_mode:
        q.train_qcnn(use_callback=True)
    else:

        tq.test_qcnn(save=save)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--train', dest='train_mode', action='store_true', help='Set this flag to train the model. Default is to test.')
    parser.add_argument('--test', dest='train_mode', action='store_false', help='Set this flag to test the model. This is the default mode.')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits. Default is 4.')
    parser.add_argument('--bit_num', type=int, default=4, help='Number of bits. Default is 4.')
    parser.add_argument('--snr_train', type=int, default=10, help='SNR for training. Default is 10.')

    parser.set_defaults(train_mode=True)

    # Parse arguments
    args = parser.parse_args()

    # Update global variables
    num_qubits = args.num_qubits
    bit_num = args.bit_num
    snr_train = args.snr_train

    # Call main with the appropriate arguments
    main_qcnn(train_mode=args.train_mode, num_qubits=args.num_qubits, bit_num=args.bit_num,
              snr_train=args.snr_train)