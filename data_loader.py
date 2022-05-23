from torch.utils.data import DataLoader, Dataset
from email.policy import default
import os
from re import I
import torch
from collections import defaultdict
import logging
import numpy as np
import re 
from tqdm import tqdm
import multiprocessing as mp
from itertools import zip_longest
from data import Corpus

class Data(Dataset):
    def __init__(self, words, chars,targets,  nchars, seq_len):
        self.nchars = nchars
        self.seq_len = seq_len
        self.data_words, self.data_chars, self.data_target = self.batchify(words, chars, targets)
        
    def batchify(self, data_words, data_chars, data_target):
        nseqs = data_words.size(0) // self.seq_len # floor division

        data_words = data_words.narrow(0,0,nseqs*self.seq_len)
        data_chars = data_chars.narrow(0,0, nseqs*self.seq_len)
        data_target = data_target.narrow(0,0, nseqs*self.seq_len)
        data_words = data_words.reshape(nseqs, self.seq_len)
        data_chars = data_chars.reshape(nseqs, self.seq_len*self.nchars)
        data_target = data_target.reshape(nseqs, self.seq_len*self.nchars) 
        #assert data_words.shape[0] ==  int(data_chars.shape[0]/12), print('wrong batchifying')
        return data_words, data_chars, data_target

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_words.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return {'words': self.data_words[index], 'chars': self.data_chars[index], 'target':self.data_target[index]}

def show_matrix_with_chars(matr, dictio):
    chars = []
    for i in matr:
        chars.append(dictio[int(i)])
    print(chars)

if __name__ == '__main__':
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy'
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/CharLSTMLM/testfiles'
    seq_len = 15
    nchars = 12
    corpus = Corpus(path, nchars,seq_len)
    data = DataLoader(Data(corpus.train_words,corpus.train_chars,corpus.train_targets,nchars, seq_len), batch_size=20)
    #for batch_ndx, sample in enumerate(data):
        # get batch for word and character data
    sample = next(iter(data))
    data_word = sample['words']
    data_char = sample['chars'][:,nchars:] # has to be next word
    show_matrix_with_chars(data_char[0], corpus.dictionary.idx2char)
    data_char = data_char.reshape(-1, (seq_len-1), nchars) # @TODO check!!!!!!
    for i in range(data_char.shape[1]-1):
        show_matrix_with_chars(data_char[0][i], corpus.dictionary.idx2char) 
    print(data_char.shape)
    data_target = sample['target'][:, nchars:]
    data_target = data_target.reshape(-1, (seq_len-1), nchars) # @TODO check!!!!!!
    show_matrix_with_chars(data_target[0][0], corpus.dictionary.idx2char)
    """
    print(corp.train_words.shape, corp.train_chars.shape)
    ex_words = corp.train_words[:seq_len].tolist()
    ex_chars = corp.train_chars[0].tolist()
    print(ex_chars)

    word_list = [corp.dictionary.idx2word[int(id)] for id in ex_words]
    char_list = [corp.dictionary.idx2char[int(id)] for id in ex_chars]

    print(word_list)
    print(char_list)

    dat = Data(corp.train_words, corp.train_chars, nchars, seq_len)
    print(dat.data_words.shape, dat.data_chars.shape)
    
    ex_words = dat.data_words[-1].tolist()
    ex_chars = dat.data_chars[-1].tolist()

    word_list = [corp.dictionary.idx2word[int(id)] for id in ex_words]
    char_list = [corp.dictionary.idx2char[int(id)] for id in ex_chars]
    print(word_list)
    print(char_list) 

    """