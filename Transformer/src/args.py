import argparse


def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Transformer single model')


	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'decode'], help='Modes: train, decode')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	# parser.add_argument('-debug', action='store_true', help='Operate on debug mode')

	# Run name should just be alphabetical word (no special characters to be included)
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='count_data', help='Dataset')
	parser.add_argument('-display_freq', type=int, default=200, help='number of batches after which to display loss')
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	parser.add_argument('-add_val_vocab', dest='add_val_vocab', action='store_true', help='Add words from Validation Set to Vocabulary')
	parser.add_argument('-no-add_val_vocab', dest='add_val_vocab', action='store_false', help='Don\'t add words from Validation Set to Vocabulary')
	parser.set_defaults(add_val_vocab=False)

	parser.add_argument('-ap', dest='ap', action='store_true', help='Arithmetic Progression')
	parser.add_argument('-no-ap', dest='ap', action='store_false', help='Not Arithmetic Progression')
	parser.set_defaults(ap=False)

	parser.add_argument('-enc_dec_res', dest='enc_dec_res', action='store_true', help='Keep Encoder-Decoder Residual Connection')
	parser.add_argument('-no-enc_dec_res', dest='enc_dec_res', action='store_false', help='Remove Encoder-Decoder Residual Connection')
	parser.set_defaults(enc_dec_res=True)

	parser.add_argument('-dec_dec_res', dest='dec_dec_res', action='store_true', help='Keep Decoder-Decoder Residual Connection')
	parser.add_argument('-no-dec_dec_res', dest='dec_dec_res', action='store_false', help='Remove Decoder-Decoder Residual Connection')
	parser.set_defaults(dec_dec_res=True)

	parser.add_argument('-continuous', dest='cont', action='store_true', help='Continuous Sequence: _ given to decoder')
	parser.add_argument('-no-continuous', dest='cont', action='store_false', help='Non-continuous Sequence: <s> given to decoder')
	parser.set_defaults(cont=False)

	parser.add_argument('-src_lang', type=str, default='en', help='Source Language')
	parser.add_argument('-trg_lang', type=str, default='de', help='Target Language')

	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')

	parser.add_argument('-heads', type=int, default=1, help='Number of Attention Heads')
	parser.add_argument('-layers', type=int, default=1, help='Number of layers in each encoder and decoder')
	parser.add_argument('-d_model', type=int, default=128, help='Embedding dimensions of inputs and hidden representations (refer Vaswani et. al)')
	parser.add_argument('-d_ff', type=int, default=256, help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
	parser.add_argument('-max_length', type=int, default=60, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-dropout', type=float, default=0.0, help= 'Dropout probability for input/output/state units (0.0: no dropout)')

	parser.add_argument('-lr', type=float, default=0.0002, help='Learning rate')
	parser.add_argument('-batch_size', type=int, default=128, help='Batch size')
	parser.add_argument('-epochs', type=int, default=50, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='noam', choices=['noam','adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')


	return parser
