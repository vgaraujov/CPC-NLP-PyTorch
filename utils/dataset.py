""" Dataset loader for the BookCorpus dataset """
import torch
import torch.utils.data as data
import numpy as np
import os
import functools
import operator
import logging
from glob import glob
from utils.vocab import *

## Get the same logger from main"
logger = logging.getLogger("cpc")

class BookCorpus(data.Dataset):
    def __init__(self, config):

        """
        Args:
            config (box): hyperparameters file
            mode (string): type of dataset
        """
        # Hyperparameters
        self.books_path = config.dataset.books_path
        self.window = config.dataset.window
        self.do_lower_case = config.dataset.do_lower_case
        self.max_sen_length = config.dataset.max_sen_length
        self.vocab_size = config.dataset.vocab_size
        self.pad = config.dataset.padding_idx
        self.unk = config.dataset.unknown_idx
        # load paths of books in subfolder <data>
        logger.info("Creating features from dataset at %s", self.books_path)
        file_list = list(sorted(glob(os.path.join(self.books_path,'data', '*.txt'))))
        self.examples = []
        for file_path in file_list:
            text = [line.rstrip('\n') for line in open(file_path, encoding="utf-8")]
            for i in range(0, len(text), self.window):
                chunk = text[i:i + self.window]
                if len(chunk) == self.window:
                    self.examples.append(chunk)
        # Load Tokenizer
        logger.info("Loading Tokenizer")
        self.word_dict = build_and_save_dictionary(text=functools.reduce(operator.iconcat, self.examples, []),
                                                   source='{}/{}'.format(self.books_path, 'vocabulary_cpc'))
        self.word_dict_reversed = {v:k for k, v in self.word_dict.items()}

    def __getitem__(self, i):
        chunk = [self.convert_sentence_to_indices(example) for example in self.examples[i]]
        return np.array(chunk)

    def __len__(self):
        return len(self.examples)

    def convert_sentence_to_indices(self, sentence):
        indices = [
                      # assign an integer to each word, if the word is too rare assign unknown token
                      self.word_dict.get(w) if self.word_dict.get(w, self.vocab_size + 1) < self.vocab_size else self.unk

                      for w in sentence.split()  # split into words on spaces
                  ][: self.max_sen_length]  # take only maxlen words per sentence at the most.
        # last words are PAD
        indices += [self.pad] * (self.max_sen_length - len(indices))
        return indices
    
    def convert_indices_to_sentence(self, indices):
        sentence = ""
        for i in indices:
            if i == 0:
                sentence += "<pad>"
            elif i == 1:
                sentence += "<unk>"            
            else:
                sentence += self.word_dict_reversed.get(i)
            sentence += " "
        return sentence