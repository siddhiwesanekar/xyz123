import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
data = pickle.load( open( "/data/training_data_old", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
parent_classes = data['parent_classes']

import json
with open('data/teamcenter_data.json') as json_data:
    intents = json.load(json_data)


net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)


model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    
    sentence_words = nltk.word_tokenize(sentence)
    
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))



model.load('./model.tflearn')




ERROR_THRESHOLD = 0.60
def classify(sentence):
    
    results = model.predict([bow(sentence, words)])[0]
    
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    
    return return_list


def response_command(sentence):
    results = model.predict([bow(sentence, words)])[0]
    
    return_list = []
    for r in results:
        for i in intents['intents']:
            
            if (i['parent_tag'] == sentence):
                return_list.append(random.choice(i['patterns']))
       
    print(return_list)
    return return_list


def response1(sentence, userID='123', show_details=False):
    if 'None of the above' in sentence:
        return ['I do not know the answer to the question. Please contact Support Team']

    if sentence in parent_classes:
        results = response_command(sentence)
        
        return_list = []
        return_list.append("Specify one of the following :")

        if results:
            while results:
                for i in intents['intents']:
                    if (i['parent_tag'] == sentence):
                        return_list.append(random.choice(i['patterns']))
                    
                return_list.append('None of the above')
                return return_list
        else:
            return ("I will transfer your question to support team")




    else:
        results = classify(sentence)
        
        if results:
            while results:
                for i in intents['intents']:
                    if i['tag'] == results[0][0]:
                        return random.choice(i['responses'])

                results.pop(0)
        else:
            return ("I will transfer your question to support team")



