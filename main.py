# -- Public Imports
import argparse

# -- Private Imports
import test_QE2E as tqf
import QE2E as qf

# -- Functions


def main(train_mode=True, num_qubits=4, bit_num=4, snr_train=10, use_onehot=True, channel_type='rayleigh', retrain=False, save=True):
    """
    Main function to either train or test the model based on the provided arguments.
    """
    qf.set_global_var(num_qubits_val=num_qubits, bit_num_val=bit_num, use_onehot_val=use_onehot, channel_type_val=channel_type)
    if train_mode:
        qf.train(num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train, use_onehot=use_onehot, retrain=retrain, save=save)
    else:
        tqf.test(num_qubits=num_qubits, bit_num=bit_num, snr_train=snr_train, use_onehot=use_onehot, save=save)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--train', dest='train_mode', action='store_true', help='Set this flag to train the model. Default is to test.')
    parser.add_argument('--test', dest='train_mode', action='store_false', help='Set this flag to test the model. This is the default mode.')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits (channel_use). Default is 4.')
    parser.add_argument('--bit_num', type=int, default=4, help='Number of bits (bit_num). Default is 4.')
    parser.add_argument('--snr_train', type=int, default=10, help='SNR for training. Default is 10.')
    parser.add_argument('--save', dest='save', action='store_true', help='Set this flag to save the models. Default is not to save.')
    parser.add_argument('--onehot', dest='use_onehot', action='store_true', help='Set this flag to use onehot encoding. Default is not to use onehot.')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='Set this flag to retrain the model. Default is to test..')
    parser.add_argument('--channel_type', dest='channel_type', type=str, choices=['awgn', 'rayleigh'], required=True,  help="Specify the channel type.")
    
    parser.set_defaults(train_mode=True, save=True, use_onehot=True, channel_type='rayleigh')

    # Parse arguments
    args = parser.parse_args()

    # Call main with the appropriate arguments
    main(train_mode=args.train_mode, num_qubits=args.num_qubits, bit_num=args.bit_num,
         snr_train=args.snr_train, retrain=args.retrain, save=args.save, use_onehot=args.use_onehot,
         channel_type=args.channel_type)
