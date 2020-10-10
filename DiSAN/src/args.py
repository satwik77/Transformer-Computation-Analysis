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
	parser.add_argument('-dataset', type=str, default='X', help='Dataset')
	parser.add_argument('-display_freq', type=int, default=200, help='number of batches after which to display loss')
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	parser.add_argument('-mt', dest='mt', action='store_true', help='Machine Translation')
	parser.add_argument('-no-mt', dest='mt', action='store_false', help='Not Machine Translation')
	parser.set_defaults(mt=False)

	parser.add_argument('-ap', dest='ap', action='store_true', help='Arithmetic Progression')
	parser.add_argument('-no-ap', dest='ap', action='store_false', help='Not Arithmetic Progression')
	parser.set_defaults(ap=False)

	parser.add_argument('-continuous', dest='cont', action='store_true', help='Continuous Sequence: _ given to decoder')
	parser.add_argument('-no-continuous', dest='cont', action='store_false', help='Non-continuous Sequence: <s> given to decoder')
	parser.set_defaults(cont=False)

	parser.add_argument('-src_lang', type=str, default='en', help='Source Language')
	parser.add_argument('-trg_lang', type=str, default='de', help='Target Language')

	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')

	# Input files
	# parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
	# parser.add_argument('-res_file', type=str, default='generations.txt', help='File name to save results in')
	# parser.add_argument('-res_folder', type=str, default='Generations', help='Folder name to save results in')
	# parser.add_argument('-out_dir', type=str, default='out', help='Out Dir')
	# parser.add_argument('-len_sort', action="store_true", help='Sort based on length')


	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# Dont modify ckpt_file
	# If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
	# parser.add_argument('-ckpt', type=str, default='./models/iwslt.pt', help='Checkpoint file name')
	# parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')


	# Model parameters
	# parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-heads', type=int, default=4, help='Number of Attention Heads')
	parser.add_argument('-layers', type=int, default=3, help='Number of layers in each encoder and decoder')
	parser.add_argument('-d_model', type=int, default=512, help='Embedding dimensions of inputs and hidden representations (refer Vaswani et. al)')
	parser.add_argument('-d_ff', type=int, default=1024, help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
	# parser.add_argument('-beam_width', type=int, default=10, help='Specify the beam width for decoder')
	parser.add_argument('-max_length', type=int, default=50, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-dropout', type=float, default=0.0, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	# parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	# parser.add_argument('-bidirectional', action='store_true', help='Initialization range for seq2seq model')

	# Training parameters
	parser.add_argument('-lr', type=float, default=0.0003, help='Learning rate')
	# parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('-epochs', type=int, default=50, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='noam', choices=['noam','adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	# parser.add_argument('-tfr', type=float, default=0.9, help='Teacher forcing ratio')


	return parser
