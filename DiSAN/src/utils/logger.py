import logging
import pdb
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json

'''Logging Modules'''

#log_format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s - %(funcName)5s() ] | %(message)s'
def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	# logger.addFilter(ContextFilter(expt_name))

	return logger


def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	# string = string.strip()
	logger.info(string)



def store_results(config, bleu_score, error_score):
	#pdb.set_trace()
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}

	try:
		train_loss = train_loss.item()
	except:
		pass

	try:
		val_loss = val_loss.item()
	except:
		pass

	#try:

	data= {'run_name' : str(config.run_name)
	, 'best bleu score' : str(bleu_score)
	, 'minimum error' : str(error_score)
	, 'dataset' : config.dataset
	, 'd_model' : config.d_model
	, 'd_ff' : config.d_ff
	, 'layers' : config.layers
	, 'heads': config.heads
	, 'dropout' : config.dropout
	, 'lr' : config.lr
	, 'batch_size' : config.batch_size
	, 'epochs' : config.epochs
	}
	# res_data.update(data)
	res_data[str(config.run_name)] = data

	with open(config.result_path, 'w', encoding='utf-8') as f:
		json.dump(res_data, f, ensure_ascii= False, indent= 4)
	#except:
	#	pdb.set_trace()

def store_val_results(config, acc_score):
	#pdb.set_trace()
	try:
		with open(config.val_result_path) as f:
			res_data = json.load(f)
	except:
		res_data = {}

	try:

		data= {'run_name' : str(config.run_name)
		, 'acc score': str(acc_score)
		, 'dataset' : config.dataset
		, 'emb1_size': config.emb1_size
		, 'emb2_size': config.emb2_size
		, 'cell_type' : config.cell_type
		, 'hidden_size' : config.hidden_size
		, 'depth' : config.depth
		, 'dropout' : config.dropout
		, 'init_range' : config.init_range
		, 'bidirectional' : config.bidirectional
		, 'lr' : config.lr
		, 'batch_size' : config.batch_size
		, 'opt' : config.opt
		, 'use_word2vec' :config.use_word2vec
		}
		# res_data.update(data)
		res_data[str(config.run_name)] = data

		with open(config.val_result_path, 'w', encoding='utf-8') as f:
			json.dump(res_data, f, ensure_ascii= False, indent= 4)
	except:
		pdb.set_trace()