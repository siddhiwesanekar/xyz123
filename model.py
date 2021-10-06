import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import random
import json

bot_name = "SmartBot"
with open('teamcenter_data.json', 'r') as f:
    intents = json.load(f)


words = []
parent_classes = []
classes = []
documents = []
ignore = ['?','!','*']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        w = nltk.word_tokenize(pattern)
        
        words.extend(w)
        
        documents.append((w, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        if intent['parent_tag'] not in parent_classes:
            parent_classes.append((intent['parent_tag']))


words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))
parent_classes = sorted(list(set(parent_classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)
print (len(parent_classes), "parent classes", parent_classes)

training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)
output_empty_1 = [0] * len(parent_classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output_row_1 = list(output_empty_1)
   # output_row_1[parent_classes.index((doc[2]))] = 1
    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])


from tensorflow.python.framework import ops

ops.reset_default_graph()

#tf.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


import pickle
pickle.dump( {'words':words, 'classes':classes,'parent_classes':parent_classes , 'train_x':train_x, 'train_y':train_y}, open( "training_data_old", "wb" ) )
