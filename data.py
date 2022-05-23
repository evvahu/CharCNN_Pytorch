# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from email.policy import default
import os
from re import I
from regex import W
import torch
from collections import defaultdict
import logging
import numpy as np
import re 
from tqdm import tqdm
import multiprocessing as mp
from itertools import zip_longest
from multiprocessing import Pool
from itertools import repeat
import math
import logging 
from concurrent import futures
from collections import deque
import itertools
import pickle
def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)

class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)
        self.char2freq = defaultdict(int)
        self.char2idx = {}
        self.idx2char = []
        vocab_path = os.path.join(path, 'vocab.txt')
        char_vocab_path = os.path.join(path, 'char_vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            char_vocab = open(char_vocab_path, encoding='utf8').read()
            self.char2idx = {c:i+1 for i,c in enumerate(char_vocab.split())}
            self.idx2char = [c for c in char_vocab.split()]
            self.char2idx['*PAD*'] = 0
            self.idx2char.insert(0, '*PAD*')
            
            self.add_char('<unk>')
            self.add_char('<bow>')
            self.add_char('<eow>')
            self.idx2word = [w for w in vocab.split()]
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.add_word('<unk>')
            
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, create a vocab file first.")

    def add_char(self, char):
        self.char2freq[char] += 1
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)

    def get_char_dict(self):
        char2idx = dict()
        idx2char = ['*PAD*']
        i = 1
        max_l = 0
        for w in self.idx2word:
            if len(w) > max_l:
                max_l = len(w)
            for c in w:
                c = c.lower()
                if c not in char2idx:
                    idx2char.append(c)
                    char2idx[c] = i
                    i += 1
        
        char2idx['<eow>'] = i
        idx2char.append('<eow>')
        char2idx['<bow>'] = i + 1
        idx2char.append('<bow>')

        return char2idx, idx2char, max_l
   


class Corpus(object):
    def __init__(self, path, word_max_l, seq_l, parallel=False, cpu_count=None):

        self.max_l = word_max_l
        self.seq_l = seq_l
        if not cpu_count:
            self.cpu_count = mp.cpu_count()
        else:
            self.cpu_count = cpu_count
        self.dictionary = Dictionary(path)
        if parallel: # path, cpu_count, dictionary, max_l
            self.train_words, self.train_chars, self.train_targets = tokenize(os.path.join(path, 'train.txt'), self.cpu_count, self.dictionary, self.max_l)
            self.valid_words, self.valid_chars, self.valid_targets = tokenize(os.path.join(path, 'valid.txt'), self.cpu_count, self.dictionary, self.max_l)
            self.test_words, self.test_chars, self.test_targets = tokenize(os.path.join(path, 'test.txt'), self.cpu_count, self.dictionary, self.max_l)
        else:
            self.train_words, self.train_chars, self.train_targets = tokenize_not_parallel(os.path.join(path, 'train.txt'), self.max_l, self.dictionary)
            self.valid_words, self.valid_chars, self.valid_targets = tokenize_not_parallel(os.path.join(path, 'valid.txt'), self.max_l, self.dictionary)
            self.test_words, self.test_chars, self.test_targets = tokenize_not_parallel(os.path.join(path, 'test.txt'), self.max_l, self.dictionary)


def _grouper(self, n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def parallel_tokenize(self, chunk, dictionary, max_l):
    #words = chunk.split()
    words = []
    for sent in chunk:
        words = words + sent.split()
    ntokens = len(words)
    ids = torch.zeros(ntokens, dtype=torch.long)
    ids_chars = torch.zeros((ntokens, max_l))
    ids_chars_target = torch.zeros((ntokens, max_l))
        
    for i, w in enumerate(words):
        if w in dictionary.word2idx:
            ids[i] = dictionary.word2idx[w]
        else:
            ids[i] = dictionary.word2idx["<unk>"] 
        char_tens = torch.zeros(max_l)
        char_tens_t = torch.zeros(max_l)
        c_local = 0
        if len(w) > (max_l-2):
            w = w[:(max_l-2)]
        for c in w: 
            if c in self.dictionary.char2idx:
                char_tens[c_local+1] = dictionary.char2idx[c]
                char_tens_t[c_local] = dictionary.char2idx[c]
            else:
                char_tens[c_local+1] = dictionary.char2idx['<unk>']
                char_tens_t[c_local] = dictionary.char2idx['<unk>']
            c_local +=1 

        char_tens[0] = dictionary.char2idx['<bow>']
        char_tens_t[c_local] = dictionary.char2idx['<eow>']

        ids_chars[i:, ] = char_tens
        ids_chars_target[i:,] = char_tens_t

    return ids, ids_chars, ids_chars_target

def tokenize(self, path, cpu_count, dictionary, max_l):

    """Tokenizes a text file for training or testing to a sequence of indices format
        We assume that training and test data has <eos> symbols """
    assert os.path.exists(path)
        # Tokenize file content
    reader = open(path, 'r', encoding="utf8")
    reader_consume = open(path, 'r', encoding='utf8')
        #pool = mp.Pool(self.cpu_count)
    e = futures.ThreadPoolExecutor(max_workers=cpu_count)
        #words = torch.zeros(0)
        #chars = torch.zeros(0)
        #target = torch.zeros(0)
    words = []
    chars = []
    target = []
    logging.info('pool of {} cpus'.format(cpu_count))
    logging.info('start tokenising')
        #nr = len(list(self._grouper(100, reader)))
    size = 1000
    nr = count_iter_items(_grouper(size, reader_consume))
    print('nr: {}'.format(nr))
    for i, chunk in enumerate(_grouper(size, reader)):
            #logging.info('chunk:{}/{}'.format(i, nr))
        print('chunk:{}/{}'.format(i, nr))
            #res = pool.map(self.parallel_tokenize, chunk)
            #print(chunk[0])
        res = e.submit(parallel_tokenize, chunk, dictionary, max_l)
        words.append(res.result()[0])
        chars.append(res.result()[1])
        target.append(res.result()[2])
            #words = torch.cat((words, res.result()[0].float()))
            #chars = torch.cat((chars, res.result()[1]))
            #target = torch.cat((target, res.result()[2]))
            #for tup in res:#.result():
            #    words.append(tup[0])
            #    chars.append(tup[1])
            #    target.append(tup[2])
            #    words = torch.cat((words, tup[0].float()))
            #    chars = torch.cat((chars, tup[1]))
            #    target = torch.cat((target, tup[2]))
        #pool.close()
    e.shutdown()
    words = torch.cat(words)
    chars = torch.cat(chars)
    target = torch.cat(target)
    print('result shape', words.shape, chars.shape, target.shape)
    return words, chars, target


def tokenize_not_parallel(path, max_l, dictionary):
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        nchars = 0
        nlines = 0
        for line in f:
            words = line.split()
            ntokens += len(words)
            nlines +=1

    print('nlines: {}'.format(nlines))
    ids = torch.zeros(ntokens, dtype=torch.long)
    ids_chars = torch.zeros((ntokens, max_l))
    ids_chars_target = torch.zeros((ntokens, max_l))
    with open(path, 'r') as rf:
        for l in tqdm(rf, total=nlines):
            #print(l)
            words = l.strip().split()
            for i, w in enumerate(words):
                #print(w)
                if w in dictionary.word2idx:
                    ids[i] = dictionary.word2idx[w]
                else:
                    ids[i] = dictionary.word2idx["<unk>"] 
                    #char_tens = torch.zeros(self.max_l)
                    #char_tens_t = torch.zeros(self.max_l)
                c_local = 0

                if len(w) > (max_l-2):
                    w = w[:(max_l-2)]
                for c in w: 
                    if c in dictionary.char2idx:
                        ids_chars[i][c_local+1] = dictionary.char2idx[c]
                        ids_chars_target[i][c_local] = dictionary.char2idx[c] 
                            #char_tens[c_local+1] = self.dictionary.char2idx[c]
                            #char_tens_t[c_local] = self.dictionary.char2idx[c]
                    else:          
                        ids_chars[i][c_local+1] = dictionary.char2idx['<unk>']
                        ids_chars_target[i][c_local] = dictionary.char2idx['<unk>'] 
                            #char_tens[c_local+1] = self.dictionary.char2idx['<unk>']
                            #char_tens_t[c_local] = self.dictionary.char2idx['<unk>']
                    c_local +=1 

                ids_chars[i][0] = dictionary.char2idx['<bow>']
                ids_chars[i][c_local] = dictionary.char2idx['<eow>']

                    #ids_chars[i:, ] = char_tens
                    #ids_chars_target[i:,] = char_tens_t
    print(ids.shape, ids_chars.shape)
    return ids, ids_chars, ids_chars_target
if __name__ == '__main__':
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_splits_small'
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/CharLSTMLM/testfiles'

    wl = 12
    sl = 15
    #corp = Corpus(path, wl, sl, parallel=True)
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_short_object.pickle'
    #f = open(path, 'wb')
    #pickle.dump(corp, f)
    #f.close()
    f = open(path, 'rb')
    corpus = pickle.load(f)
    print(corpus.train_words[0:sl,])
    f.close()