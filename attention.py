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
import re
import numpy as np
import random
import os
import io
import time
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

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
def preprocess(sentence, language = 'unspecified'):

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
    if len(sentence.split(' ')) < MAX_SENTENCE_LENGTH and sentence.startswith('I'):
        return True
    return False

def readLangs(corpus, source_lang, target_lang):
    #The quick test data is tab split i.e each line is "English Sentence \t Phrase Francais"
    lines = open('corpus',encoding='utf-8').read().strip().split('\n')

    #Create the dictionaries for both languages and return the sentence pairs
    pairs = []

    source_vocab = Vocab(source_lang)
    target_vocab = Vocab(target_lang)
    
    for line in lines:
        sentences = line.split('\t')
        source =  preprocess(sentences[0], source_lang)
        target = preprocess(sentences[1], target_lang)
        if meetsDataRequirements(source):
            pairs.append([source, target])
            source_vocab.addSentence(source)
            target_vocab.addSentence(target)

    print('Read Data: Source Unique Words: %d Target Unique Words: %d' % (source_vocab.n_words, target_vocab.n_words))
    
    return source_vocab, target_vocab, pairs
    

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
        '''
        Useful wrapper class for saving models as one package instead of having multiple files for a single experiment.
        '''
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.learning_rate = learning_rate

        #Optimizers and criterion can have large impact on performance
        #We set them to be Stochastic Gradient Descent and Negative Log Likelihood
        #Though might be useful to be overridable to experiment with different values.
        self.encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()


    def train_iteration(self, source_tensor,target_tensor, max_sentence_length = MAX_SENTENCE_LENGTH):
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

    def train(self, X, max_iters, source_lang, target_lang, print_rate=1000, plot_rate=100):
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
            source_tensor = source_lang.sentenceTensor(x[0])
            target_tensor = target_lang.sentenceTensor(x[1])

            loss = self.train_iteration(source_tensor, target_tensor)

            loss_since_print += loss
            loss_since_plot += loss
            
            if i % print_rate == 0 and i != 0:
                average_loss = float(loss_since_print) / print_rate
                loss_since_print = 0
                print('%d %.4f' % (i, average_loss))
            
            if i % plot_rate == 0 and i != 0:
                average_loss = float(loss_since_plot) / plot_rate
                losses.append(average_loss)
                loss_since_plot = 0
        
        return losses

    def predict(self, input, source_lang, target_lang, max_sentence_length = MAX_SENTENCE_LENGTH):
        '''
        Given the trained encooder decoder, predict the target lang version of the input sentence.
        '''
        decoded_output = []
        source_tensor = source_lang.sentenceTensor(input)

        source_length = source_tensor.size(0)

        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(max_sentence_length, self.encoder.hidden_size, device = device)

            for ei in range(source_length):
                encoder_output , encoder_hidden = self.encoder(source_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[START_TOKEN_INDEX]], device=device)
            decoder_hidden = encoder_hidden

            for di in range(max_sentence_length):
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
                decoded_output.append(target_lang.index2word[decoder_input.item()])

            # print(decoded_output)            
        return decoded_output

    def get_bleu(self, X, source_lang, target_lang):
        predictions = []
        targets = []
        for x in X:
            source = x[0]
            #Might need to detokenize here...
            targets.append(x[1])
            predictions.append(' '.join(self.predict(source, source_lang, target_lang)))

        targets = [targets]
        print(targets[:10])
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

def baseline(label, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
    encoder = EncoderRNN(src_vocab_B.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_B.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses = translator.train(train_X_B, iters_B, src_vocab_B, target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, source_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_baseline_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'
    with open(filename, 'wb') as f:
        pickle.dump(losses, f)

    filename = '%s/translator_final.dump'
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    return losses, validation_bleu, translator

def transfer_all(label, src_vocab_A, target_vocab_A, X_A, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
    encoder = EncoderRNN(src_vocab_A.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_A.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses_A = translator.train(X_A, iters_A, src_vocab_A, target_vocab_A)

    losses_B = translator.train(train_X_B, iters_B, src_vocab_B, target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, source_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_all_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump'
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    return losses, validation_bleu, translator
    

def transfer_decoder(label, src_vocab_A, target_vocab_A, X_A, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder, initiailze the attention and decoder to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
    #PreTrain
    encoder = EncoderRNN(src_vocab_A.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_A.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses_A = translator.train(X_A, iters_A, src_vocab_A, target_vocab_A)

    #Transfer
    encoder = EncoderRNN(src_vocab_B.n_words, hidden_size).to(device)
    translator = Translator(encoder, decoder, attention)

    losses_B = translator.train(train_X_B, iters_B, src_vocab_B, target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, source_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_decoder_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump'
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    return losses_B, validation_bleu, translator

def transfer_attention(label, src_vocab_A, target_vocab_A, X_A, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder and decoder, initiailze the attention to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
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

    validation_bleu = translator.get_bleu(test_X_B, source_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_attention_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump'
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    return losses_B, validation_bleu, translator
def transfer_encoder_attention(label, src_vocab_A, target_vocab_A, X_A, src_vocab_B, target_vocab_B, train_X_B, test_X_B, iters_A, iters_B):
    '''
    Pretrain the encoder decoder translator on language set A (i.e English - Hungarian)
    Create a new model for the encoder and decoder, initiailze the attention to be the same as the trained attention.
    Then continue training full model on language set B (.e English - Finnish)

    The translator for set B is the desired translator.

    Returns:
     - the negative log likelihood at each iteration of training in the second stage 
     - the bleu score found from the validation set provided.
     - the translator itself.
    '''
    
    hidden_size = 256
    #PreTrain
    encoder = EncoderRNN(src_vocab_A.n_words, hidden_size).to(device)
    attention = BahdanauAttention(hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, target_vocab_A.n_words, attention).to(device)

    translator = Translator(encoder, decoder, attention)
    losses_A = translator.train(X_A, iters_A, src_vocab_A, target_vocab_A)

    #Transfer
    decoder = DecoderRNN(hidden_size, target_vocab_B.n_words, attention).to(device)
    translator = Translator(encoder, decoder, attention)

    losses_B = translator.train(train_X_B, iters_B, src_vocab_B, target_vocab_B)    

    validation_bleu = translator.get_bleu(test_X_B, source_vocab_B, target_vocab_B)
    #Clumsly make a local directory.
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = './%s_encoder_%s' % (label, dt)
    os.mkdir(dir_name)

    filename = '%s/losses.dump'
    with open(filename, 'wb') as f:
        pickle.dump(losses_B, f)

    filename = '%s/translator_final.dump'
    with open(filename, 'wb') as f:
        pickle.dump(translator, f)

    return losses_B, validation_bleu, translator


def run_experiments(pretrain_iterations, test_iterations):
    #Hungarian
    src_H, target_H, X_H = readLangs('en-hu.txt', 'en', 'hu')
    #Finnish
    src_F, target_F, X_F = readLangs('en-fn.txt', 'en', 'fn')
    #Estonian
    src_E, target_E, X_E = readLangs('en-es.txt', 'en', 'es')
    #Swedish
    src_E, target_E, X_E = readLangs('en-sw.txt', 'en', 'sw')
    #Slovak
    src_E, target_E, X_E = readLangs('en-sk.txt', 'en', 'sk')
    #Czech
    src_E, target_E, X_E = readLangs('en-cz.txt', 'en', 'cz')

    #Hungarian/Finnish
    src_CH, target_CH, X_CH = readComboLangs('en-hu.txt','en-fn.txt' 'en', 'hu-fn')

    #Swedish/Finnish
    src_CH, target_CH, X_CH = readComboLangs('en-sw.txt','en-fn.txt' 'en', 'sw-fn')

    #Czech/Hungarian
    src_CH, target_CH, X_CH = readComboLangs('en-hu.txt','en-cz.txt' 'en', 'cz-hu')


    # Finnish Translator Exps.

    # BASELINE EN - FI
    
    # GENETIC EN - HU => EN - FI

    # REGIONAL EN - SW => EN - FI

    # COMBO GENETIC EN - HU/FI => EN - FI

    # COMBO REGIONAL EN - SW/FI => EN -FI

    # Hungarian Translator Exps.

    # BASELINE EN - HU
    
    # GENETIC EN - FI => EN - HU

    # REGIONAL EN - CZ => EN - HU

    # COMBO GENETIC EN - HU/FI => EN - HU

    # COMBO REGIONAL EN - CZ/HU => EN - HU

def run_estonian_test():
    #Estionain data set is particularly small, so would be interesting to use this as a final test.


if __name__ == "__main__":
    # src, target, X = readLangs()
    # hidden_size = 256
    # encoder = EncoderRNN(src.n_words, hidden_size).to(device)
    # attention = BahdanauAttention(hidden_size).to(device)
    # decoder = DecoderRNN(hidden_size, target.n_words, attention).to(device)
    # translator = Translator(encoder, decoder, attention)
    # losses = translator.train(X, 75000, src, target, print_rate = 1000, plot_rate=500)

    # save_model(encoder, 'encoder.p')
    # save_model(decoder, 'decoder.p')
    # save_model(attention, 'attention.p')
    # english_sentence = 'I am cold'
    # print(translator.predict(english_sentence, src, target))

    # #This is just to test if the code works, actually need to properly remove a test set from training data.
    # test_test_set = X[:100]
    # print(translator.get_bleu(test_test_set, src, target))
    
    # fig, ax = plt.subplots()
    # ax.plot(losses)
    # ax.set(ylabel='Average Negative Log Likelihood')
    # fig.savefig('loss.png')

