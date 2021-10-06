import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
data = pickle.load( open( "training_data_old", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
parent_classes = data['parent_classes']
# import our chat-bot intents file
import json
with open('teamcenter_data.json') as json_data:
    intents = json.load(json_data)


net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
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
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response_command(sentence):
    results = model.predict([bow(sentence, words)])[0]
    # print(results)
    # filter out predictions below a threshold
    # results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    # results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        for i in intents['intents']:
            # if i['tag'] == r[0][0]:
            if (i['parent_tag'] == sentence):
                return_list.append(random.choice(i['patterns']))
        # print(r)
        # print(classes[r[0]])
        # print(parent_classes[r[0]])
        # return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    print(return_list)
    return return_list


def response(sentence, userID='123', show_details=False):
    if 'None of the above' in sentence:
        return ['I do not know the answer to the question. Please contact Support Team']

    if sentence in parent_classes:
        results = response_command(sentence)
        # print(results)
        return_list = []
        return_list.append("Specify one of the following :")

        if results:
            while results:
                for i in intents['intents']:
                    if (i['parent_tag'] == sentence):
                        return_list.append(random.choice(i['patterns']))
                    #print(return_list)
                return_list.append('None of the above')
                return return_list
        else:
            return ("I will transfer your question to support team")




    else:
        results = classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            while results:
                for i in intents['intents']:
                    if i['tag'] == results[0][0]:
                        return random.choice(i['responses'])

                results.pop(0)
        else:
            return ("I will transfer your question to support team")
