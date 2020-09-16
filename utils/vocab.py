"""
This code has been taken and modified from https://github.com/ryankiros/skip-thoughts
Constructing and loading dictionaries
"""
import pickle as pkl
from collections import OrderedDict


def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    worddict['<pad>'] = 0
    worddict['<unk>'] = 1
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx+2 # 0: <pad>, 1: <unk>

    return worddict, wordcount


def load_dictionary(loc='./data/book_dictionary_large.pkl'):
    """
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
        
    return worddict


def save_dictionary(worddict, wordcount, loc='./data/book_dictionary_large.pkl'):
    """
    Save a dictionary to the specified location
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
#         pkl.dump(wordcount, f)


def build_and_save_dictionary(text, save_loc, vocab_size):
    try:
        cached = load_dictionary(save_loc)
        print("Using cached dictionary at {}".format(save_loc))
        return cached
    except:
        pass
    # build again and save
    print("Unable to load from cached, building fresh")
    worddict, wordcount = build_dictionary(text)
    print("Got {} unique words".format(len(worddict)))
    # fixing to vocab_size
    if vocab_size:
        worddict = OrderedDict(list(worddict.items())[:vocab_size])
    print("Saving dictionary at {}".format(save_loc))
    save_dictionary(worddict, wordcount, save_loc)
    
    return worddict