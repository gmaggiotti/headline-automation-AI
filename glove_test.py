# -*- coding: utf-8 -*-
# # Word Embedding (Word2Vec)
# ## read GloVe

glove_name = "data-es/glove/SBW-vectors-100-200k.txt"
import commands
cmd_result =commands.getstatusoutput('wc -l '+glove_name)
glove_n_symbols = int(cmd_result[1].split()[0])
embedding_dim = 100

#   get glove word2vec into an array
import numpy as np
glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale=.1
with open(glove_name, 'r') as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = map(float,l[1:])
        i += 1

### glove_index_dict[ idx_autoinc]
### glove_embedding_weights has the embd vectors
glove_embedding_weights *= globale_scale
glove_embedding_weights.std()

reverse_dictionary = dict((idx,word) for word,idx in glove_index_dict.iteritems())


word_vec = glove_embedding_weights[glove_index_dict['más']]
sim = np.dot(word_vec, -glove_embedding_weights.T).argsort()[0:8]
for idx in range(8):
    print reverse_dictionary[sim[idx]]

