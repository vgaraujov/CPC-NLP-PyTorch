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
        """
        # Hyperparameters
        self.books_path = config.dataset.books_path
        self.vocab_name = config.dataset.vocab_name
        self.window = config.dataset.window
        self.do_lower_case = config.dataset.do_lower_case
        self.max_sen_length = config.dataset.max_sen_length
        self.pad = config.dataset.padding_idx
        self.unk = config.dataset.unknown_idx
        # load paths of books in subfolder <data>
        logger.info("Creating features from dataset at %s", self.books_path)
        file_list = list(sorted(glob(os.path.join(self.books_path,'data', '*.txt'))))
        self.examples = []
        for file_path in file_list:
            text = [line.lower().rstrip('\n') if self.do_lower_case else line.rstrip('\n') 
                    for line in open(file_path, encoding="utf-8")]
            for i in range(0, len(text), self.window):
                chunk = text[i:i + self.window]
                if len(chunk) == self.window:
                    self.examples.append(chunk)
        # Load Tokenizer
        logger.info("Loading vocabulary")
        self.word_dict = build_and_save_dictionary(text=functools.reduce(operator.iconcat, self.examples, []),
                                                   source='{}/{}'.format(self.books_path, self.vocab_name), 
                                                   vocab_size=config.dataset.vocab_size)
        self.word_dict_reversed = {v:k for k, v in self.word_dict.items()}

    def __getitem__(self, i):
        chunk = [self.convert_sentence_to_indices(example) for example in self.examples[i]]
        return np.array(chunk)

    def __len__(self):
        return len(self.examples)

    def convert_sentence_to_indices(self, sentence):
        indices = [
                      # assign an integer to each word, if the word is too rare assign unknown token
                      self.word_dict.get(w,self.unk) for w in sentence.split()  # split into words on spaces
                  ][: self.max_sen_length]  # take only maxlen words per sentence at the most.
        # last words are PAD
        indices += [self.pad] * (self.max_sen_length - len(indices))
        return indices
    
    def convert_indices_to_sentence(self, indices):
        sentence = ""
        for i in indices:
            sentence += self.word_dict_reversed.get(int(i))
            sentence += " "
        return sentence
    
    
class SentimentAnalysis(data.Dataset):
    def __init__(self, config, mode):

        """
        Args:
            config (box): hyperparameters file
            mode (string): type of dataset (train, dev or test)
        """
        # Hyperparameters
        self.dataset_path = config.sentiment_analysis.dataset_path
        self.dataset_name = config.sentiment_analysis.dataset_name
        self.vocab_file_path = config.dataset_classifier.vocab_file_path
        self.do_lower_case = config.dataset_classifier.do_lower_case
        self.max_sen_length = config.dataset_classifier.max_sen_length
        self.pad = config.dataset_classifier.padding_idx
        self.unk = config.dataset_classifier.unknown_idx
        # Load Data
        full_path = '{}/data/{}/{}.{}.txt'.format(self.dataset_path, self.dataset_name, self.dataset_name.lower(), mode)
        raw_data = [line.rstrip('\n') for line in open(full_path)]
        self.data = [line.split(' ||| ')[1].strip() for line in raw_data]
        self.label = [line.split(' ||| ')[0] for line in raw_data]
        self.classes = len(set(self.label))
        # Load Tokenizer
        print('Loading vocabulary')
        try:
            self.word_dict = load_dictionary(self.vocab_file_path)
            self.word_dict_reversed = {v:k for k, v in self.word_dict.items()}
        except:
            print('You must have vocabulary dictionary of the BookCorpus dataset')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
        """
        sentence = self.data[index]
        if self.do_lower_case:
            sentence = sentence.lower()
        ids = self.convert_sentence_to_indices(sentence)
        label = int(self.label[index])
  
        return np.array(ids), label
        
    def __len__(self):
        return len(self.data)
    
    def convert_sentence_to_indices(self, sentence):
        indices = [
                      # assign an integer to each word, if the word is too rare assign unknown token
                      self.word_dict.get(w,self.unk) for w in sentence.split()  # split into words on spaces
                  ][: self.max_sen_length]  # take only maxlen words per sentence at the most.
        # last words are PAD
        indices += [self.pad] * (self.max_sen_length - len(indices))
        return indices
        
    def convert_indices_to_sentence(self, indices):
        sentence = ""
        for i in indices:
            sentence += self.word_dict_reversed.get(int(i))
            sentence += " "
        return sentence