"""
This code has been taken and modified from https://github.com/ryankiros/skip-thoughts
Vocabulary Expansion
"""
import gensim
import itertools
import collections
import sklearn.linear_model
import torch
import torch.nn as nn
import numpy as np
from box import box_from_file
from pathlib import Path
from timeit import default_timer as timer

## Custom Imports
from model.models import CPCv1
from utils.vocab import *


def _expand_vocabulary(skip_thoughts_emb, skip_thoughts_vocab, word2vec):
    """Runs vocabulary expansion on a skip-thoughts model using a word2vec model.
    Args:
    skip_thoughts_emb: A numpy array of shape [skip_thoughts_vocab_size,
        skip_thoughts_embedding_dim].
    skip_thoughts_vocab: A dictionary of word to id.
    word2vec: An instance of gensim.models.Word2Vec.
    Returns:
    combined_emb: A dictionary mapping words to embedding vectors.
    """
    # Find words shared between the two vocabularies.
    print("Finding shared words")
    shared_words = [w for w in word2vec.vocab if w in skip_thoughts_vocab]

    # Select embedding vectors for shared words.
    print("Selecting embeddings for {} shared words".format(len(shared_words)))
    shared_st_emb = skip_thoughts_emb[[
      skip_thoughts_vocab[w] for w in shared_words
    ]]
    shared_w2v_emb = word2vec[shared_words]

    # Train a linear regression model on the shared embedding vectors.
    print("Training linear regression model")
    model = sklearn.linear_model.LinearRegression()
    model.fit(shared_w2v_emb, shared_st_emb)

    # Create the expanded vocabulary.
    print("Creating embeddings of expanded vocabulary")
    combined_emb = collections.OrderedDict()
    for w in word2vec.vocab:
        # Ignore words with underscores (spaces).
        if "_" not in w:
            w_emb = model.predict(word2vec[w].reshape(1, -1))
            combined_emb[w] = w_emb.reshape(-1)

    for w in skip_thoughts_vocab:
        combined_emb[w] = skip_thoughts_emb[skip_thoughts_vocab[w]]

    print("Created expanded vocabulary of {} words".format(len(combined_emb)))
        
    return combined_emb

def save_expansion(embedding_map, config):
    embeddings=[]
    vocab_dict = collections.OrderedDict()
    for idx, (word, emb) in enumerate(embedding_map.items()):
        vector = np.array(emb, dtype='float32')
        embeddings.append(vector)
        vocab_dict[word] = int(idx)
    embeddings = np.array(embeddings)
    assert embeddings.shape[0] == len(embedding_map)
    # saving expanded files
    np.save('vocab_expansion/embeddings_expanded.npy', embeddings)
    with open('vocab_expansion/vocab_expanded.pkl', 'wb') as f:
        pkl.dump(vocab_dict, f)
    
def main(run_name, word2vec_path):
    config = box_from_file(Path('config_cpc.yaml'), file_type='yaml')
    use_cuda = False # use CPU
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use_cuda is', use_cuda)
    # load pretrained model
    print("Loading pretrained CPC model: {}".format(run_name))
    cpc_model = CPCv1(config=config)
    checkpoint = torch.load('{}/{}-{}'.format(config.training.logging_dir, run_name,'model_best.pth'))
    cpc_model.load_state_dict(checkpoint['state_dict'])
    cpc_model.to(device)
    # get lookup table
    cpc_model.eval()
    output = cpc_model.get_word_embedding(torch.arange(config.dataset.vocab_size).to(device))
    skip_thoughts_emb = output.detach().cpu().numpy()

    # load original vocab dictionary
    print("Loading CPC dictionary")
    skip_thoughts_vocab = load_dictionary(loc='vocab.pkl')
    assert len(skip_thoughts_vocab) == config.dataset.vocab_size
    
    # Load the Word2Vec model
    print('Loading word2vec vectors at {}'.format(word2vec_path))
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # Run vocabulary expansion
    embedding_map = _expand_vocabulary(skip_thoughts_emb, skip_thoughts_vocab, word2vec)
    # Save expanded embeddings and dictionary
    print("Saving expanded embeddings and vocabulary")
    save_expansion(embedding_map, config)
    
if __name__ == "__main__":
    #-----------------------------------------------------------------------------#
    # Specify model and dictionary locations here
    #-----------------------------------------------------------------------------#
    run_name = "cpc-2020-09-13_12_32_03"
    word2vec_path = "word2vec/GoogleNews-vectors-negative300.bin.gz"
    Path("vocab_expansion").mkdir(parents=True, exist_ok=True)
    
    global_timer = timer() # global timer
    main(run_name, word2vec_path)
    end_global_timer = timer()
    print("Total elapsed time: {}".format(end_global_timer - global_timer))