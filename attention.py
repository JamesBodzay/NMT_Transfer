'''
The code written here is an adaptation of the tutorial provided by PyTorch for
Neural Machine Translation with Attention
'''
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import sacrebleu
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
#import re
import regex as re
import numpy as np
import random
import os
import io
import time
import datetime
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_TOKEN_INDEX = 0
END_TOKEN_INDEX = 1
SPACE_TOKEN_INDEX = 2
START_TOKEN = "SOS"
END_TOKEN = "EOS"
SPACE_TOKEN = " "

MAX_SENTENCE_LENGTH=10
MAX_SENTENCE_LENGTH_CHAR=60

'''
Represenetation of language which contains dictionaries to be able to translate between vector representation and 
string representation

Possible ToDos
----
 - Train and use a word2vec model on all languages ( with supplemental data ) to use in place of this structure
 - Segment words into smaller tokens, represent words as combination of those vectors.
'''
class Vocab:
    def __init__(self, name):
        self.name = name
        self.n_words = 2
        self.word2index = {}
        self.word2count = {}
        self.index2word = {START_TOKEN_INDEX: START_TOKEN, END_TOKEN_INDEX: END_TOKEN}

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def sentenceIndices(self, input):
        return [self.word2index[word] for word in input.split(' ')]

    def sentenceTensor(self, input):
        indices = self.sentenceIndices(input)
        indices.append(END_TOKEN_INDEX)
        return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Alphabet:
    '''
    Representation of the possible characters for all languages under test 
    '''
    def __init__(self):
        self.n_chars = 2
        self.char2index = {}
        self.char2count = {}
        self.index2char = {START_TOKEN_INDEX: START_TOKEN, END_TOKEN_INDEX: END_TOKEN}

    def addSentence(self, sentence):
        for char in sentence:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def sentenceIndices(self, input):
        return [self.char2index[char] for char in input]

    def sentenceTensor(self, input):
        indices = self.sentenceIndices(input)
        indices.append(END_TOKEN_INDEX)
        return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)


'''
This is just required for the quick test data. Actual data sets used may differ
'''
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


'''
Preprocess sequence of words given language
'''
def preprocess(sentence, language = 'unspecified', use_diacritics = True):

    # print("in: %s" %(sentence))
    if use_diacritics:
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence  = re.sub(r"[^\p{L}]+", r" ", sentence)
    else:
        sentence = unicodeToAscii(sentence.lower().strip())
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence  = re.sub(r"[^a-zA-Z]+", r" ", sentence)

    # print("out: %s" %(sentence))

    return sentence

'''
Only accept a subset of the data to make training times shorter.
'''
def meetsDataRequirements(sentence):
    if len(sentence.split(' ')) < MAX_SENTENCE_LENGTH and len(sentence) < MAX_SENTENCE_LENGTH_CHAR:
        return True
    return False

def readLangs(corpus, source_lang, target_lang, alphabet = None, source_vocab = None):
    '''
    Read lines from corpus file into the source and target language.
    If an alphabet has been passed, add characters to that alphabet in order to build up  a communal alphabet
    If not, create a new alphabet.
    Returns:
    - the source and target vocab of words
    - the updated alphabet
    - sentence pairs for the two languages (which can be used as input data to the models)
    '''

    #The data is tab split i.e each line is "English Sentence \t Phrase Francais"
    lines = open(corpus,encoding='utf-8').read().strip().split('\n')

    #Create the dictionaries for both languages and return the sentence pairs
    pairs = []

    target_vocab = Vocab(target_lang)

    if alphabet is None:
        alphabet = Alphabet()

    if source_vocab is None:
        source_vocab = Vocab(source_lang)
    
    # print(lines)
    count = 0
    for line in lines:
        sentences = line.split('\t')
        source =  preprocess(sentences[0], source_lang)
        target = preprocess(sentences[1], target_lang)
        if meetsDataRequirements(source):
            count += 1
            pairs.append([source, target])
            source_vocab.addSentence(source)
            target_vocab.addSentence(target)
            alphabet.addSentence(source)
            alphabet.addSentence(target)

    print('Read %s: Source Unique Words: %d Target Unique Words: %d Alphabet Unique Chars: %d Num Lines: %d' % (corpus,
                                                                                                                source_vocab.n_words,
                                                                                                                target_vocab.n_words,
                                                                                                                alphabet.n_chars,
                                                                                                                count))
    return source_vocab, target_vocab, alphabet, pairs

def readComboLangs(corpusA,corpusB, source_lang, target_lang, alphabet = None, source_vocab = None,max_of_a=10000, max_of_b=1000):
    '''
    Read lines from two corpus files into the source and target language.
    Assumes the source is shared between the two languages
    If an alphabet has been passed, add characters to that alphabet in order to build up  a communal alphabet
    If not, create a new alphabet.
    Adds of up to max_of_b lines from the second corpus.

    Returns:
    - the source and target vocab of words
    - the updated alphabet
    - sentence pairs for the two languages (which can be used as input data to the models)
    '''


    #The data is tab split i.e each line is "English Sentence \t Phrase Francais"
    lines = open(corpusA,encoding='utf-8').read().strip().split('\n')


    target_vocab = Vocab(target_lang)

    if alphabet is None:
        alphabet = Alphabet()

    if source_vocab is None:
        source_vocab = Vocab(source_lang)
    
    # print(lines)
    count = 0
    a_pairs = []
    for line in lines:
        sentences = line.split('\t')
        source =  preprocess(sentences[0], source_lang)
        target = preprocess(sentences[1], target_lang)
        if meetsDataRequirements(source):
            count += 1
            a_pairs.append([source, target])
            source_vocab.addSentence(source)
            target_vocab.addSentence(target)
            alphabet.addSentence(source)
            alphabet.addSentence(target)


    lines = open(corpusB,encoding='utf-8').read().strip().split('\n')
    b_pairs = []
    for line in lines:
        sentences = line.split('\t')
        source =  preprocess(sentences[0], source_lang)
        target = preprocess(sentences[1], target_lang)
        if meetsDataRequirements(source):
            count += 1
            b_pairs.append([source, target])
            source_vocab.addSentence(source)
            target_vocab.addSentence(target)
            alphabet.addSentence(source)
            alphabet.addSentence(target)


    pairs =random.sample(a_pairs, min(max_of_a, len(a_pairs)))
    pairs = pairs + (random.sample(b_pairs, min(max_of_b, len(b_pairs))))
    print('Read %s / %s: Source Unique Words: %d Target Unique Words: %d Alphabet Unique Chars: %d Num Lines: %d' % (corpusA,corpusB,
                                                                                                                source_vocab.n_words,
                                                                                                                target_vocab.n_words,
                                                                                                                alphabet.n_chars,
                                                                                                                count))
    return source_vocab, target_vocab, alphabet, pairs
    

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
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
    
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
        scores = scores.squeeze(2).unsqueeze(1)
        attention_weights = func.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, values.unsqueeze(0))
        self.attention_weights = attention_weights

        return attention_weights, context



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
            attention_weights , context = self.attention(hidden, encoder_outputs, encoder_outputs)
            merged_vector = torch.cat((embedded, context), 2)
            output = func.relu(merged_vector)
            # print(output.size())
        else:
            output = func.relu(embedded)
        output, hidden = self.gru(output, hidden, bidirectional=True)
        output = func.log_softmax(self.out(output[0]), dim = 1)
        return output, hidden, attention_weights  

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)      

class Translator():

    def __init__(self, encoder, decoder, attention = None, learning_rate = 0.01, max_sentence_length = MAX_SENTENCE_LENGTH):
        '''
        Useful wrapper class for saving models as one package instead of having multiple files for a single experiment.
        '''
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.learning_rate = learning_rate
        self.max_sentence_length = max_sentence_length

        #Optimizers and criterion can have large impact on performance
        #We set them to be Stochastic Gradient Descent and Negative Log Likelihood
        #Though might be useful to be overridable to experiment with different values.
        self.encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()


    def train_iteration(self, source_tensor,target_tensor):
        '''
        Using one source/target sentence perform one step of encoder and decoder optimization
        Applys the encoder and decoder optimizers on the loss function determined by
        the provided criterion

        Because we train one sentence at a time, we can potentially transfer language information from one to another
        by first training all 'A' languages then training all 'B' sentences.
        Might need to reset the decoder weights. Encoder/Attention weights could probably be kept the same.
        '''
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        source_length = source_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_sentence_length, self.encoder.hidden_size, device = device)

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

    def train(self, X, max_iters, source_lang = None, target_lang = None, alphabet = None, print_rate=1000, plot_rate=100):
        '''
        Train the encoder decoder on the given input data.
        Prints the average loss every $print_rate iterations.
        Plots the average loss every $plot_rate iterations
        '''
        loss_since_print = 0
        loss_since_plot = 0

        X_rand = [random.choice(X) for i in range(max_iters)]
        losses = []
        for i, x in enumerate(X_rand):
            # print(x)
            if alphabet is not None:
                source_tensor = alphabet.sentenceTensor(x[0])
                target_tensor = alphabet.sentenceTensor(x[1])
            elif source_lang is not None and target_lang is not None:
                source_tensor = source_lang.sentenceTensor(x[0])
                target_tensor = target_lang.sentenceTensor(x[1])
            else:
                print('No vocabulary or alphabet provided. Cannot create tensors from sentences.')
                break

            loss = self.train_iteration(source_tensor, target_tensor)

            loss_since_print += loss
            loss_since_plot += loss
            
            if i % print_rate == 0 and i != 0:
                average_loss = float(loss_since_print) / print_rate
                loss_since_print = 0
                print('%d %.4f' % (i, average_loss))
            
            if i % plot_rate == 0:
                if i != 0:
                    average_loss = float(loss_since_plot) / plot_rate
                else:
                    average_loss = float(loss_since_plot)
                losses.append(average_loss)
                loss_since_plot = 0
        
        return losses

    def predict(self, input, source_lang = None, target_lang = None, alphabet = None):
        '''
        Given the trained encooder decoder, predict the target lang version of the input sentence.

        Returns the prediction
        '''
        decoded_output = []
        if alphabet is not None:
            source_tensor = alphabet.sentenceTensor(input)
        elif source_lang is not None and target_lang is not None:
            source_tensor = source_lang.sentenceTensor(input)
        else:
            print('No vocabulary or alphabet provided. Cannot create tensors from sentences.')
            return None
        source_length = source_tensor.size(0)

        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_sentence_length, self.encoder.hidden_size, device = device)

            for ei in range(source_length):
                encoder_output , encoder_hidden = self.encoder(source_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[START_TOKEN_INDEX]], device=device)
            decoder_hidden = encoder_hidden

            for di in range(self.max_sentence_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Don't have access to the actual target sentence here, so we use the most likely prediction as the input
                #to the next word.
                topv, topi = decoder_output.topk(1)
                # print (topv)
                # print(topi)
                #Get the index in the embedding in the target language
                decoder_input = topi.squeeze().detach()

                if decoder_input.item() == END_TOKEN_INDEX:
                    break
                if alphabet is not None:
                    decoded_output.append(alphabet.index2char[decoder_input.item()])
                elif target_lang is not None:
                    decoded_output.append(target_lang.index2word[decoder_input.item()])
                else:
                    print('No alphabet or target vocab provided. Unable to de-embed output.')
            # print(decoded_output)            
        return decoded_output

    def get_bleu(self, X, source_lang=None, target_lang=None,alphabet = None):
        predictions = []
        targets = []
        for x in X:
            source = x[0]
            #Might need to detokenize here...
            targets.append(x[1])
            if alphabet is not None:
                #If translation is character based then spaces will be included in the translation
                predictions.append(''.join(self.predict(source, alphabet=alphabet)))
            elif source_lang is not None and target_lang is not None:
                #If translation is word based, add space inbetween words in translation
                predictions.append(' '.join(self.predict(source, source_lang, target_lang)))
            else:
                print('No alphabet or vocabulary provided, unable to create predictions.')

        targets = [targets]
        # print(targets[:10])
        print(predictions[:10])
        bleu = sacrebleu.corpus_bleu(predictions, targets)
        return bleu.score



def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def baseline_word(label, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_B):
    '''
    Train the encoder decoder translator on language set B (i.e English - Hungarian)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
    print("Experiment: Baseline [Word]")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, 0, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH))
    encoder = EncoderRNN(src_vocab_B.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_B.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses = translator.train(train_X_B, iters_B, source_lang=src_vocab_B, target_lang=target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, source_lang = src_vocab_B, target_lang = target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_baseline_word_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses, f)

    filename = '%s/translator_final.dump'  % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses[-1], validation_bleu))
    print("==========")
    return losses, validation_bleu, translator

def baseline_char(label, alphabet, train_X_B, test_X_B, iters_B):
    '''
    Train the encoder decoder translator on language set B (i.e English - Hungarian)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 64
    print("Experiment: Baseline [Char]")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, 0, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))

    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention=attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses = translator.train(train_X_B, iters_B, alphabet=alphabet)    

    validation_bleu = translator.get_bleu(test_X_B, alphabet=alphabet)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_baseline_char_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'  % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses, f)

    filename = '%s/translator_final.dump'  % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses[-1], validation_bleu))
    print("==========")
    return losses, validation_bleu, translator

def transfer_all(label, alphabet, X_A, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    As different languages will have different embeddings on a word based level.
    This method can only be used for character based methods.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 64
    print("Experiment: Transfer All")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses_A = translator.train(X_A, iters_A, alphabet=alphabet)

    losses_B = translator.train(train_X_B, iters_B, alphabet=alphabet)    

    validation_bleu = translator.get_bleu(test_X_B, alphabet=alphabet)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_all_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'  % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump'  % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator
    

def transfer_decoder(label, alphabet, X_A, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder, initiailze the attention and decoder to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    As different languages will have different embeddings on a word based level.
    This method can only be used for character based methods.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 64
    print("Experiment: Transfer Decoder and Attention")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))
    #PreTrain
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses_A = translator.train(X_A, iters_A, alphabet=alphabet)

    #Transfer
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)

    losses_B = translator.train(train_X_B, iters_B,alphabet=alphabet)    

    validation_bleu = translator.get_bleu(test_X_B, alphabet=alphabet)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_decoder_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator

def transfer_attention_word(label, src_vocab_A, target_vocab_A, X_A, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder and decoder, initiailze the attention to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    Allows for the source language for each set to be different. However it is unlikely this will achieve meaningul results.

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256

    print("Experiment: Transfer Attention [Word]")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH))
    #PreTrain
    encoder = EncoderRNN(src_vocab_A.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_A.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses_A = translator.train(X_A, iters_A, src_vocab_A, target_vocab_A)

    #Transfer
    encoder = EncoderRNN(src_vocab_B.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_B.n_words, attention).to(device)
    translator = Translator(encoder, decoder, attention)

    losses_B = translator.train(train_X_B, iters_B, src_vocab_B, target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, src_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_attention_word_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator

def transfer_attention_char(label, alphabet, X_A, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder and decoder, initiailze the attention to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Uses the character based translation method instead of word based.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 64
    print("Experiment: Transfer Attention [CHAR]")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))
    #PreTrain
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses_A = translator.train(X_A, iters_A,alphabet = alphabet)

    #Transfer
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)
    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)

    losses_B = translator.train(train_X_B, iters_B, alphabet = alphabet)    

    validation_bleu = translator.get_bleu(test_X_B, alphabet = alphabet)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_attention_char_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)
    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator

#If going to transfer encoder/decoder, the encoder and decoder need to be the same dimensionality.
#Further, we want the embeddings to be the same, otherwise we are relying on having a translation between multiple embeddings,
#which creates a circular problem of needing a translator to create a translator.
def transfer_encoder_attention_char(label, alphabet, X_A, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the  decoder, initiailze the attention and encoder to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    This method is for character based models. For word embeddings, use transfer_encoder_attention_word()

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    hidden_size = 64

    print("Experiment: Transfer Encoder and Attention")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))
    #PreTrain
    encoder = EncoderRNN(alphabet.n_chars, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses_A = translator.train(X_A, iters_A, alphabet = alphabet)

    #Transfer
    decoder = DecoderRNN(hidden_size, alphabet.n_chars, attention).to(device)
    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)

    losses_B = translator.train(train_X_B, iters_B,alphabet=alphabet)    

    validation_bleu = translator.get_bleu(test_X_B, alphabet=alphabet)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_encoder_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator

def transfer_encoder_attention_word(label, src_vocab_A, target_vocab_A, X_A, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the decoder, initiailze the attention encoder to be the same as the trained attention.
    Assumes that the source language for both sets is the same and has the same embedding.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    This method is for word embeddings, for character based models use transfer_encoder_attention_char

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    hidden_size = 256

    print("Experiment: Transfer Encoder and Attention")
    print("Label: %s PreTrain Iters: %d FinalTrain Iters: %d" % (label, iters_A, iters_B))
    print("Hidden Size: %d Max Sentence Length: %d" % (hidden_size, MAX_SENTENCE_LENGTH_CHAR))
    #PreTrain
    encoder = EncoderRNN(src_vocab_A.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_A.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)
    losses_A = translator.train(X_A, iters_A, source_lang=src_vocab_A, target_lang=target_vocab_A)

    #Transfer
    decoder = DecoderRNN(hidden_size, target_vocab_B.n_words, attention).to(device)
    translator = Translator(encoder, decoder, attention, max_sentence_length=MAX_SENTENCE_LENGTH_CHAR)

    losses_B = translator.train(train_X_B, iters_B,source_lang=src_vocab_A, target_lang = target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B,source_lang=src_vocab_A, target_lang=target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_encoder_word_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump' % (dir_name)
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    print("Final Loss: %f Validation Bleu: %d" % (losses_B[-1], validation_bleu))
    print("==========")
    return losses_B, validation_bleu, translator


def run_model_experiments(pretrain_iterations, test_iterations):
    #Hungarian
    src, target_H, alph, X_H = readLangs('en-hu.txt', 'en', 'hu')
    #Finnish
    src, target_F, alph, X_F = readLangs('en-fn.txt', 'en', 'fn', alphabet=alph, source_vocab=src)
    
    train_X_F, test_X_F = train_test_split(X_F, test_size = 0.2, train_size=0.8)

    #Word Based
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig, ax = plt.subplots()

    losses, _ , _ = baseline_word('model_test', src, target_F,train_X_F, test_X_F, test_iterations)
    #Increment the x axis in steps of the default plot rate. This is a bit of a quick hack.
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    losses, _, _ = transfer_attention_word('model_test', src, target_H, X_H,src, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Attention")

    losses, _, _ = transfer_encoder_attention_word('model_test', src, target_H, X_H, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Encoder and Attention")

    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Word Based Model Training Error')
    ax.legend()
    fig.savefig("word_based_models_%s.png" % (dt))

    #Char based

    fig, ax = plt.subplots()

    losses, _, _ = baseline_char('model_test', alph, train_X_F, test_X_F, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    losses, _, _ = transfer_all('model_test', alph, X_H, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Whole Model")

    losses, _, _ = transfer_encoder_attention_char('model_test', alph, X_H, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Encoder and Attention")

    losses, _, _ = transfer_decoder('model_test', alph, X_H, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Decoder and Attention")

    losses, _, _ = transfer_attention_char('model_test', alph, X_H, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Pretrained Attention")

    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Char Based Model Training Error')
    ax.legend()
    fig.savefig("char_based_models_%s.png" % (dt))


def run_lang_char_experiments(pretrain_iterations, test_iterations):
    #Ensure that the alphabet and source vocab is shared amongst all language pairs
    #That way we can transfer knowledge between them.
    #Hungarian
    src, target_H, alph, X_H = readLangs('en-hu.txt', 'en', 'hu')
    #Finnish
    src, target_F, alph, X_F = readLangs('en-fn.txt', 'en', 'fn', alphabet=alph, source_vocab=src)
    #Estonian
    src, target_E, alph, X_E = readLangs('en-es.txt', 'en', 'es', alphabet=alph, source_vocab=src)
    #Swedish
    src, target_SW, alph, X_SW = readLangs('en-sw.txt', 'en', 'sw', alphabet=alph, source_vocab=src)
    #Slovak
    src, target_SK, alph, X_SK = readLangs('en-sk.txt', 'en', 'sk', alphabet=alph, source_vocab=src)
    #Czech
    src, target_C, alph, X_C = readLangs('en-cz.txt', 'en', 'cz', alphabet=alph, source_vocab=src)

    #Hungarian/Finnish
    src, target_HF, alph, X_HF = readComboLangs('en-hu.txt','en-fn.txt', 'en', 'hu-fn', 
                                                alphabet=alph, source_vocab=src, 
                                                max_of_a=(10*test_iterations), max_of_b=test_iterations)
    src, target_FH, alph, X_FH = readComboLangs('en-fn.txt','en-hu.txt', 'en', 'fn-hu',
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    #Swedish/Finnish
    src, target_SF, alph, X_SF = readComboLangs('en-sw.txt','en-fn.txt', 'en', 'sw-fn',
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    #Czech/Hungarian
    src, target_CH, alph, X_CH = readComboLangs('en-cz.txt','en-hu.txt', 'en', 'cz-hu', 
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    train_X_H, test_X_H = train_test_split(X_H, test_size = 0.2, train_size=0.8)

    train_X_F, test_X_F = train_test_split(X_F, test_size = 0.2, train_size=0.8)

    # Finnish Translator Exps.

    # BASELINE EN - FI
    fig, ax = plt.subplots()
    losses, _ , _  = baseline_char('en_fi', alph, train_X_F, test_X_F, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    # GENETIC EN - HU => EN - FI
    losses, _ , _  = transfer_all('hu-fi',alph, X_H, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Hungarian")
    # REGIONAL EN - SW => EN - FI
    losses, _ , _  = transfer_all('sw-fi',alph, X_SW, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Swedish")
    # COMBO GENETIC EN - HU/FI => EN - FI
    losses, _ , _  = transfer_all('combo-hu-fi',alph, X_HF, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Hungarian-Finnish")

    # COMBO REGIONAL EN - SW/FI => EN -FI
    losses, _ , _  = transfer_all('combo-sw-fi',alph, X_SF, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Swedish-Finnish")
    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Char Based Language Transfer Training Error')
    ax.legend()
    fig.savefig("finnish-lang-char.png")

    # Hungarian Translator Exps.
    fig, ax = plt.subplots()
    # BASELINE EN - HU
    losses, _ , _  = baseline_char('en_hu', alph, train_X, test_X, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    # GENETIC EN - FI => EN - HU
    losses, _ , _  = transfer_all('fi-hu',alph, X_F, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Finnish")

    # REGIONAL EN - CZ => EN - HU
    losses, _ , _  = transfer_all('cz-hu',alph, X_C, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Czech")

    # COMBO GENETIC EN - HU/FI => EN - HU
    losses, _ , _  = transfer_all('combo-fi-hu',alph, X_FH, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Finnish-Hungarian")

    # COMBO REGIONAL EN - CZ/HU => EN - HU
    losses, _ , _  = transfer_all('combo-cz-hu',alph, X_CH, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Czech-Hungarian")
    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Char Based Language Transfer Training Error')
    ax.legend()
    fig.savefig("hungarian-lang-char.png")

def run_lang_word_experiments(pretrain_iterations, test_iterations):
    #Ensure that the alphabet and source vocab is shared amongst all language pairs
    #That way we can transfer knowledge between them.
    #Hungarian
    src, target_H, alph, X_H = readLangs('en-hu.txt', 'en', 'hu')
    #Finnish
    src, target_F, alph, X_F = readLangs('en-fn.txt', 'en', 'fn', alphabet=alph, source_vocab=src)
    #Swedish
    src, target_SW, alph, X_SW = readLangs('en-sw.txt', 'en', 'sw', alphabet=alph, source_vocab=src)
    #Czech
    src, target_C, alph, X_C = readLangs('en-cz.txt', 'en', 'cz', alphabet=alph, source_vocab=src)

    #Hungarian/Finnish
    src, target_HF, alph, X_HF = readComboLangs('en-hu.txt','en-fn.txt', 'en', 'hu-fn', a
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)
    src, target_FH, alph, X_FH = readComboLangs('en-fn.txt','en-hu.txt', 'en', 'fn-hu', 
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    #Swedish/Finnish
    src, target_SF, alph, X_SF = readComboLangs('en-sw.txt','en-fn.txt', 'en', 'sw-fn',
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    #Czech/Hungarian
    src, target_CH, alph, X_CH = readComboLangs('en-cz.txt','en-hu.txt', 'en', 'cz-hu', 
                                                alphabet=alph, source_vocab=src,
                                                max_of_a=(10*test_iterations,) max_of_b=test_iterations)

    train_X_H, test_X_H = train_test_split(X_H, test_size = 0.2, train_size=0.8)

    train_X_F, test_X_F = train_test_split(X_F, test_size = 0.2, train_size=0.8)

    # Finnish Translator Exps.

    # BASELINE EN - FI
    losses, _, _ = baseline_word('en_fi', src, target_F, train_X_F, test_X_F, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    # GENETIC EN - HU => EN - FI
    losses, _, _ = transfer_encoder_attention_word('hu-fi', src, target_H, X_H, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Hungarian")
    # REGIONAL EN - SW => EN - FI
    losses, _, _ = transfer_encoder_attention_word('sw-fi', src, target_SW, X_SW, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Swedish")
    # COMBO GENETIC EN - HU/FI => EN - FI
    losses, _, _ = transfer_encoder_attention_word('combo_hu-fi', src, target_HF, X_HF, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Hungarian-Finnish")
    # COMBO REGIONAL EN - SW/FI => EN -FI
    losses, _, _ = transfer_encoder_attention_word('combo_sw-fi', src, target_SF, X_SF, target_F, train_X_F, test_X_F, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Swedish-Finnish")
    
    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Word Based Language Transfer Training Error')
    ax.legend()
    fig.savefig("finnish-lang-word.png")

    # Hungarian Translator Exps.

    # BASELINE EN - HU
    losses, _, _ = baseline_word('en_hu', src, target_H, train_X_H, test_X_H, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Baseline")

    # GENETIC EN - FI => EN - HU
    losses, _, _ = transfer_encoder_attention_word('fi-hu', src, target_F, X_F, target_H, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Finnish")
    # REGIONAL EN - CZ => EN - HU
    losses, _, _ = transfer_encoder_attention_word('cz-hu', src, target_C, X_C, target_H, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Czech")
    # COMBO GENETIC EN - HU/FI => EN - HU
    losses, _, _ = transfer_encoder_attention_word('combo_fi_hu', src, target_FH, X_FH, target_H, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Finnish-Hungarian")
    # COMBO REGIONAL EN - CZ/HU => EN - HU
    losses, _, _ = transfer_encoder_attention_word('combo_cz_hu', src, target_CH, X_CH, target_H, train_X_H, test_X_H, pretrain_iterations, test_iterations)
    ax.plot(np.arange(0,test_iterations,100), losses, label="Combined Czech-Hungarian")

    ax.set(xlabel='Epoch', ylabel='Average Negative Log Likelihood', title='Word Based Language Transfer Training Error')
    ax.legend()
    fig.savefig("hungarian-lang-word.png")


def quick_test():
    '''
    For performing quick dev trials outside of the main method.
    '''
    hidden_size = 64
    src_H, target_H, alph, X = readLangs('en-hu.txt', 'en', 'hu')
    # src_F, target_F, alph, X = readLangs('en-fn.txt', 'en', 'fn')

    print(X[0])

    train_X, test_X = train_test_split(X, test_size = 0.2, train_size=0.8)

    print(train_X[0])

    baseline_char('quick_test', alph, train_X, test_X, 1000)


if __name__ == "__main__":

    # quick_test()
    run_model_experiments(50000, 5000)


