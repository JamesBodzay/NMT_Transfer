'''
The code written here is an adaptation of the tutorial provided by PyTorch for
Neural Machine Translation with Attention
'''
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import random
import os
import io
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
'''
Preprocess sequence of words given language
'''
def preprocess(sentence, language = 'unspecified'):
    if language is 'unspecified':
        pass
    elif langage is 'en':
        pass
    elif language is 'hu':
        pass
    return sentence

START_TOKEN_INDEX = 0
END_TOKEN_INDEX = 1
START_TOKEN = "SOS"
END_TOKEN = "EOS"

MAX_SENTENCE_LENGTH=10

'''
Represenetation of language which contains dictionaries to be able to translate between vector representation and 
string representation

Possible ToDos
----
 - Train and use a word2vec model on all languages ( with supplemental data ) to use in place of this structure
 - Segment words into smaller tokens, represent words as combination of those vectors.
'''
class Lang:
    def __init__(self, name):
        self.name = name
        self.n_words = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = {START_TOKEN_INDEX: START_TOKEN, END_TOKEN_INDEX: END_TOKEN}

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def sentenceIndices(self, input):
        return [self.word2index[word] for word in input.split(' ')]

    def sentenceTensor(self, input):
        return torch.tensor(self.sentenceIndices(input), dtype=torch.long, device=device).view(-1, 1)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

'''
This is just required for the quick test data. Actual data sets used may differ
'''
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFS', s)
        if unicodedata.category(c) != 'Mn'
    )


'''
Preprocess sequence of words given language
'''
def preprocessor(sentence, language = 'unspecified'):

    sentence = unicodeToAscii(sentence.lower().strip())
    #Remove non alphabet characters This needs some consideration for languages with diacritics
    sentence = re.sub(r"{[.!?])", r" \1", s)
    sentence  = re.sub(r"[^a-zA-Z.!?]+", r" ", s)


    #Language specific processing
    if language is 'unspecified':
        pass
    elif langage is 'en':
        pass
    elif language is 'hu':
        pass
    elif language is 'ro':
        pass
    elif language is 'fi':
        pass

    return sentence

'''
Only accept a subset of the data to make training times shorter.
'''
def meetsDataRequirements(sentence):
    if len(sentence.split(' ')) <= MAX_SENTENCE_LENGTH and sentence.startswith('I'):
        return True
    return False

def readLangs():
    #The quick test data is tab split i.e each line is "English Sentence \t Phrase Francais"
    lines = open('en-fr.txt',encoding='utf-8').read().strip().split('\n')

    #Create the dictionaries for both languages and return the sentence pairs
    pairs = []

    source_lang = Lang('en')
    target_lang = Lang('fr')
    
    for line in lines:
        sentences = line.split('\t')
        source =  preprocess(sentences[0])
        target = preprocess(sentences[1])
        if meetsDataRequirements(source):
            pairs.append([source, target])
            source_lang.addSentence(source)
            target_lang.addSentence(target)

    print('Read Data: Source Unique Words: %d Target Unique Words: %d' % (source_lang.n_words, target_lang.n_words))
    
    return source_lang, target_lang, pairs
    

'''
Implement an Encoder which inherits from the default EncoderRNN provided by torch.nn.Module
This implementation replaces the forward step with a GRU network

This is a uni directional encoder. Some improvement might be made by having a bi-directional encoder instead.
'''
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        #ToDo: Consider making input_size parameterizable.
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
    
    '''
    Forward step achieved by running the GRU layer 
    '''
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, key_size = None, query_size = None):
        super(BahdanauAttention, self).__init__()
        
        if key_size is None:
            key_size = hidden_size
        if query_size is None:
            query_size = hidden_size

        print('Query Size: %d Key_size: %d Hidden Size %d' % (query_size, key_size, hidden_size))
        self.key_linear = nn.Linear(key_size, hidden_size, bias = False)
        self.query_linear = nn.Linear(query_size, hidden_size, bias = False)
        self.score_linear = nn.Linear(hidden_size, 1, bias=False)

        #Attention scores
        self.attention_weights = 0

    def forward(self, query, key, values):
        #values = encoder_outputs
        #query = hidden
        #key = encoder outputs
        projected_query = self.query_linear(query)
        projected_key = self.key_linear(key)
        scores = self.score_linear(torch.tanh(projected_query + projected_key))
        # print(scores.size())
        scores = scores.squeeze(2).unsqueeze(1)
        attention_weights = func.softmax(scores, dim=-1)
        # print(attention_weights.size())
        # print(values.unsqueeze(0).size())
        context = torch.bmm(attention_weights, values.unsqueeze(0))
        self.attention_weights = attention_weights


        return attention_weights, context


'''
Implement a simple Decoder which uses a Gated Recurrent Unit
'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attention = None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if attention is not None:
            self.gru = nn.GRU(hidden_size * 2, hidden_size)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attention = attention
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attention_weights = None
        if self.attention is not None:
            attention_weights , context = attention(hidden, encoder_outputs, encoder_outputs)
            merged_vector = torch.cat((embedded, context), 2)
            output = func.relu(merged_vector)
            # print(output.size())
        else:
            output = func.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = func.log_softmax(self.out(output[0]), dim = 1)
        return output, hidden, attention_weights  

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)      

class Translator():

    def __init__(self, encoder, decoder, attention = None, learning_rate = 0.01):
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.learning_rate = learning_rate

        #Optimizers and criterion can have large impace on performance
        #We set them to be Stochastic Gradient Descent and Negative Log Likelihood
        #Though might be useful to be overridable to experiment with different values.
        self.encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

    '''
    Using one source/target sentence perform one step of encoder and decoder optimization
    Applys the encoder and decoder optimizers on the loss function determined by
    the provided criterion

    Because we train one sentence at a time, we can potentially transfer language information from one to another
    by first training all 'A' languages then training all 'B' sentences.
    Might need to reset the decoder weights. Encoder/Attention weights could probably be kept the same.
    '''
    def train_iteration(self, source_tensor,target_tensor, max_sentence_length = MAX_SENTENCE_LENGTH):

        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        source_length = source_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_sentence_length, self.encoder.hidden_size, device = device)

        loss = 0

        for ei in range(source_length):
            encoder_output , encoder_hidden = self.encoder(source_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[START_TOKEN_INDEX]], device=device)
        decoder_hidden = encoder_hidden

        #Uses Teacher forcing: i.e. use the actual target tensor as next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += self.criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] 

        loss.backward() #compute gradient 

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    '''
    Train the encoder decoder on the given input data.
    Prints the average loss every $print_rate iterations.
    Plots the average loss every $plot_rate iterations (ToDo)
    '''
    def train(self, X, max_iters, source_lang, target_lang, print_rate=1000, plot_rate=100):

        #Use Negative Log Likelihood as loss function

        #ToDo: choose x from X uniformly at random
        loss_since_print = 0
        loss_since_plot = 0

        X_rand = [random.choice(X) for i in range(max_iters)]
        for i, x in enumerate(X_rand):
            source_tensor = source_lang.sentenceTensor(x[0])
            target_tensor = target_lang.sentenceTensor(x[1])

            loss = self.train_iteration(source_tensor, target_tensor)

            loss_since_print += loss
            loss_since_plot += loss
            
            if i % print_rate == 0 and i != 0:
                average_loss = float(loss_since_print) / print_rate
                loss_since_print = 0
                print('%d %.4f' % (i, average_loss))


if __name__ == "__main__":
    src, target, X = readLangs()
    hidden_size = 256
    encoder = EncoderRNN(src.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target.n_words, attention).to(device)
    translator = Translator(encoder, decoder, attention)
    translator.train(X, 40000, src, target)