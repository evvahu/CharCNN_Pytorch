from xml.dom.pulldom import parseString
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import adabound
import os, copy, pickle

import numpy as np
import pandas as pd

import regex as re
import random

from model import *
from BatchLoader import *
from Optimizer import *
from DataGenerator import *
import logging
import argparse
import toml
from torch.utils.data import DataLoader
import time 
from data import Corpus, Dictionary
from data_loader import Data
import multiprocessing as mp
'''
Takes in preprocessed data stored in an class object named 
'dataprocessor'
--attributes
	.src  #input [batch_size, sequence_length, characters]
	.tgt  #label [batch_size, sequence_length]
	.max_word_length #max length of a word in the vocabulary
	.w2n # word to integer mapping dictionary
	.n2w # integer to word mapping dictionary
	.c2n # character to integer mapping dictionary
	.n2c # integer to charcter mapping dictionary

'''

def evaluate(data, model, loss_compute):
	model.eval()
	val_perplexity = 0
	total_val_loss = 0
	for batch_ndx, sample in enumerate(data):
		data_target = sample['words'][:,1:] # has to be next word 
		data_char = sample['chars'] # has to be current word 
		data_char = data_char.reshape(-1, config['seq_len'], config['word_length'])

		val_out = model(data_char)
		val_loss = loss_compute.loss_fn(val_out.transpose(1, 2), data_target)
		val_perplexity += float(torch.exp(val_loss))
		total_val_loss += float(val_loss)
	total_val_loss /= batch_ndx
	val_perplexity /= batch_ndx
	return total_val_loss, val_perplexity


def train(data, model, loss_compute):
	model.train()
	total_loss = 0
	for batch_ndx, sample in enumerate(data):
        # get batch for word and character data
		data_target = sample['words'][:,1:].long() # has to be next word @TODO check and padding
		 
		data_char = sample['chars'].long() # has to be current word 
		data_char = data_char.reshape(-1, config['seq_len'], config['word_length'])
		#print('shapes: {}, {}'.format(data_char.shape, data_target.shape))
		model.zero_grad()
		out = model(data_char)
		loss_one_step = loss_compute(out, data_target)
		total_loss += float(loss_one_step)
		print(batch_ndx, loss_one_step)
	return total_loss/batch_ndx
		
if __name__ == '__main__':
	argp = argparse.ArgumentParser()
	argp.add_argument('config_path')
	args = argp.parse_args()
	config = toml.load(args.config_path)
	logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(config['log_path'])])
	logging.info(config)
	try:
		os.makedirs(config['model_dir'])
	except:
		print('model directory already exists')
	model_path = os.path.join(config['model_dir'], 'lstmmain')

	logging.info('model path: {}'.format(model_path))
	logging.info('start loading corpus')

	if config['pickled']:
		reader = open(config['path'], 'rb')
		corpus = pickle.load(reader)
		reader.close()
	else:
		if config['parallel']:
			logging.info('cpu count:{}'.format(mp.cpu_count()))
			corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=mp.cpu_count())
		else:
			corpus = Corpus(config['path'], config['word_length'], config['seq_len'], parallel=False, cpu_count = 0)	
	logging.info('finished loading corpus')
	data_loader_train = DataLoader(Data(corpus.train_words, corpus.train_chars, corpus.train_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
	data_loader_valid = DataLoader(Data(corpus.valid_words, corpus.valid_chars,corpus.valid_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
	data_loader_test = DataLoader(Data(corpus.test_words, corpus.test_chars, corpus.test_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
    # corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=2)
	logging.info('finished with data loader')
	seq_len = config['seq_len']
	word_length = config['word_length']
    
    
	gpu = False
	if torch.cuda.is_available():
		gpu = True
	model = CharacterRNNLM(word_length, config['embedding_dim'], len(corpus.dictionary.char2idx), len(corpus.dictionary.word2idx),
	config['padding_idx'], config['kernels'], config['num_filter'], rnn_type='LSTM', num_layers = config['num_rnn_layers'], 
	d_rnn = config['d_rnn'],  dropout = config['dropout'],  bidirectional = config['bidirectional']) 

	if gpu:
		model.cuda()

	softmax = torch.nn.Softmax(dim=1)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = adabound.AdaBound(model.parameters(),  lr=1e-4, final_lr=0.01)
	scheduler = None
	Opt = SpecOptimizer(model, optimizer, scheduler, initial_lr= config['lr'], max_norm=5)
	loss_fn = nn.NLLLoss(ignore_index = config['padding_idx'])#padding_idx)
	LossCompute = LossComputer(Opt, loss_fn)
	
	eval_batch_size = config['bs']
	epochs = config['nr_epochs']

	eow = int(corpus.dictionary.char2idx['<eow>'])
    #print(eow)
	logging.info('start train')
	try:
		best_val_loss = None
		for epoch in range(1, epochs+1):
			epoch_start_time = time.time()
			train(data_loader_train, model, LossCompute)
			val_loss, val_perpl = evaluate(data_loader_valid, model, LossCompute)
			LossCompute.update(val_perpl)
			logging.info('-' * 89)
			logging.info('after epoch {}: validation loss: {}, perplexity: {},  time: {:5.2f}s'.format(epoch, val_loss, val_perpl, time.time() - epoch_start_time))
			logging.info('-' * 89)
            #print('VAL LOSS', val_loss)
			if not best_val_loss or val_loss < best_val_loss:
				with open(model_path, 'wb') as f:
					torch.save(model, f)
					best_val_loss = val_loss
			else:
				logging.info('after epoch {}, no improvement lr {} will be reduced'.format(epoch, str(lr)))

	except KeyboardInterrupt:	
		logging.info('-' * 89)
		logging.info('Exiting from training early')
        # Load the best saved model.
		with open(model_path, 'rb') as f:
			model = torch.load(f)

    # Run on test data.
	test_loss = evaluate(data_loader_test)
	logging.info('*' * 89)
	logging.info('End of training: average word probability: {}, test loss: {}'.format(0, test_loss))
	logging.info('*' * 89)



