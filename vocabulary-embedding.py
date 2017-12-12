# -*- coding: utf-8 -*-

# Generate intial word embedding for headlines and description

# The embedding is limited to a fixed vocabulary size (`vocab_size`) but
# a vocabulary of all the words that appeared in the data is built.
import re


FN = 'vocabulary-embedding'
seed=42
vocab_size = 40000
embedding_dim = 300
lower = False # dont lower case the text


# # read tokenized headlines and descriptions
import cPickle as pickle
FN0 = 'tokens' # this is the name of the data file which I assume you already have
with open('data-es/tn/sports-50k.pkl', 'rb') as fp:
    heads, desc, keywords = pickle.load(fp) # keywords are not used in this project



if lower:
    heads = [h.lower() for h in heads]
if lower:
    desc = [h.lower() for h in desc]

import HTMLParser
def polish_sentence( sentence ):
    p = HTMLParser.HTMLParser()
    sentence = p.unescape(unicode(sentence, "utf-8"))
    sentence = re.sub(u'\n','', sentence)
    sentence = re.sub(u'<[^>]*>','', sentence)
    sentence = re.sub(u'\[[a-z\_]*embed:.*\]','', sentence)
    sentence = re.sub(u'\[video:.*\]','', sentence)
    sentence = re.sub(u'[\.\[\]\?\,\(\)\!\"\'\\/\:\-]',' ', sentence)
    sentence = re.sub(u'[ ]+',' ', sentence)

# h = html2text.HTML2Text()
    # h.ignore_links = True
    # sentence = h.handle(unicode(sentence, "utf-8"))
    # del h
    #
    # sentence = re.sub('&ntilde;', "ñ",sentence)
    # sentence = re.sub(u'\n', "",sentence)
    # sentence = re.sub(u'\[social_embed:.*\]',"",sentence)
    # sentence = re.sub(u'[a-zA-Z]\.', " . ",sentence)
    #
    # spcl_chr = re.escape('[]?,()!"\'\\/:-')
    # regex = '[' + spcl_chr + ']'
    # regex_array = ['<[^>]*>',regex]
    # for i in range(regex_array.__len__()):
    #     sentence = re.sub(regex_array[i]," ", sentence)
    return sentence

# # build vocabulary
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in polish_sentence(txt).split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount


vocab, vocabcount = get_vocab(heads+desc)


# most popular tokens
print vocab[:50]
print '...',len(vocab)



empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word


# In[22]:


def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos

    idx2word = dict((idx,word) for word,idx in word2idx.iteritems())

    return word2idx, idx2word

#   this gets and index number for each word and the other back entry
#   word2idx['the']=45 => idx2word[45]=['the']
word2idx, idx2word = get_idx(vocab, vocabcount)


# # Word Embedding (Word2Vec)
# ## read GloVe

glove_name = "data-es/glove/SBW-vectors-300-1MM.txt"
import commands
cmd_result =commands.getstatusoutput('wc -l '+glove_name)
glove_n_symbols = int(cmd_result[1].split()[0])


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


# lower tokens
for w,i in glove_index_dict.iteritems():
    w1 = w.lower()
    if w1 not in glove_index_dict:
        glove_index_dict[w] = i


# ## embedding matrix
# calculate toke size
#vocab_size =idx2word.__len__()

# use GloVe to initialize embedding matrix

# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print 'random-embedding/glove scale', scale, 'std', embedding.std()

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'): # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1
print 'number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size)


# lots of word in the full vocabulary (word2idx) are outside `vocab_size`.
# Build an alterantive which will map them to their closest match in glove but only if the match
# is good enough (cos distance above `glove_thr`)



glove_thr = 0.5




word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g


# for every word outside the embedding matrix find the closest word inside the mebedding matrix.
# Use cos distance of GloVe vectors.
# 
# Allow for the last `nb_unknown_words` words inside the embedding matrix to be considered to be outside.
# Dont accept distances below `glove_thr`



normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.iteritems():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove :
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print '# of glove substitutes found', len(glove_match)


# manually check that the worst substitutions we are going to do are good enough
for orig, sub, score in glove_match[-10:]:
    print score, orig,'=>', idx2word[sub]


# build a lookup table of index of outside words to index of inside words
glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)
Y = [[word2idx[token] for token in polish_sentence(headline).split()] for headline in heads]
X = [[word2idx[token] for token in polish_sentence(d).split()] for d in desc]


import cPickle as pickle
with open('data-es/%s.pkl'%FN,'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)


import cPickle as pickle
with open('data-es/%s.data.pkl'%FN,'wb') as fp:
    pickle.dump((X,Y),fp,-1)

