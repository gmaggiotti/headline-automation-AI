# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np

import cPickle as pickle
with open('data-es/vocabulary-embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)

### Nearest 8 neighbors

def similarity(word):
    word_vec = embedding[word2idx[word]]
    sim = np.dot(word_vec, -embedding.T).argsort()[0:7]
    for idx in range(len(sim)):
        print idx2word[sim[idx]]

similarity('roja')