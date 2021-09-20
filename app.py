import tensorflow as tf
import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import random
import json

bot_name = "SmartBot"
with open('data/teamcenter_data.json', 'r') as f:
    intents = json.load(f)


words = []
parent_classes = []
classes = []
documents = []
ignore = ['?','!','*']
# loop through each sentence in the intent's patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each and every word in the sentence
        w = nltk.word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word(s) to documents
        documents.append((w, intent['tag']))
        # add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        if intent['parent_tag'] not in parent_classes:
            parent_classes.append((intent['parent_tag']))


# Perform stemming and lower each word as well as remove duplicates
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
pickle.dump( {'words':words, 'classes':classes,'parent_classes':parent_classes , 'train_x':train_x, 'train_y':train_y}, open( "data/training_data_old", "wb" ) )



import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
data = pickle.load( open( "data/training_data_old", "rb" ) )
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


import socket
import threading


from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./templates')

app.static_folder = './static'
app.temp_dict = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    msg5 = response1(userText)
    if userText.isnumeric():
        if int(userText) in app.temp_dict.keys():
            userText = app.temp_dict[int(userText)]
            userText = ''.join(userText)

    if 'form' in msg5:
        import webbrowser

        webbrowser.open("D:/PLM Nordic/UOM_request_form.docx")
    if '/' in msg5:
        if 'open' in userText:
            import webbrowser

            webbrowser.open(msg5)

            return "File opened successfully!"


        if 'delete' in userText:
            import os
            if os.path.exists(msg5):
                os.remove(msg5)
                msg5 = "File Deleted Successfully!"
                return msg5

            else:
                msg5 = "File Dose Not Exist!"
                return msg5
    if (type(msg5) is list) == True:


        res = [''.join(ele) for ele in msg5]

        temp_num = 0
        for i in res[1:]:
            temp_num = temp_num + 1
            app.temp_dict[temp_num] = []

            app.temp_dict[temp_num].append(str(i))

        return app.temp_dict



    return str(response1(userText))


conversation = []


def func():
    t = threading.Thread(target=get_bot_response)
    t.start()


attempt = 0
s = socket.socket()







if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80)
