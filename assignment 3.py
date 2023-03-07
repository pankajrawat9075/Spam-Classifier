#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
from random import randrange



# Get the dataset :  the data set is available and it contains multiple emails with labels of spam or ham on each email
data = open('SpamDetectionData.txt', 'r').read()
data = data.split('\n')

# getting the emails into the proper format
emails = [] # this list stores all the emails in text format
labels = [] # this list stores the corresponding email label spam 1 and ham 0
for line in data:
    if line[0:4] == 'Spam':
        labels.append(1)
        emails.append(line[5:])
        pass
    elif line[0:3] == 'Ham':
        labels.append(0)
        emails.append(line[4:])
        pass
    else:
        pass

print("total no. of samples:", len(labels))
print("total no. of spam samples:", labels.count(1))
print("total no. of ham samples:", labels.count(0))

# dividing the data into training and testing set, selecting randomly the data

train_emails = list()
train_labels = list()
train_size = 0.8 * len(labels)
emails_ = emails
labels_ = labels
while len(train_emails) < train_size:
    index = randrange(len(emails_))
    train_emails.append(emails_.pop(index))
    train_labels.append(labels_.pop(index))
test_labels = labels_
test_emails = emails_

# preprossesing data to get vector of numbers instead of text emails, also removing less important features

from sklearn.feature_extraction.text import TfidfVectorizer

# this vectorizer is used to convert the text emails to a vector of numbers corresponding to each word
vectorizer = TfidfVectorizer(
    input='content',     # input is actual text
    lowercase=True,      # convert to lower case before tokenizing
    stop_words='english', # remove stop words( most common used english words)
)
emails_train = vectorizer.fit_transform(train_emails)
words = vectorizer.get_feature_names_out()
emails_train = emails_train.toarray() # it represents vector of numbers representing words frequecy
emails_test  = vectorizer.transform(test_emails).toarray()

# training the model on the training dataset

trainSize = len(emails_train)
testSize = len(emails_test)
print("Train size ", trainSize)
print("Test size ", testSize)

# naive bayes classifier training
p = train_labels.count(1)/trainSize # this is probab. of y = 1 spam
# y = 1, y = 0

print("probability of an email being a spam = ", p)

lenWords = len(emails_train[0])

wordProb = np.zeros(shape=(2, lenWords)) # probab. of jth word occuring given y

Ycount = []
Ycount.append(train_labels.count(0)) # count of ham 0 in the train data
Ycount.append(trainSize - Ycount[0]) # count of sham 1 in the train data 

print("count of ham in train data ", Ycount[0])
print("count of ham in train data ", Ycount[1])

# estimating the most likelihood parameters
for y in range(2):
    for j in range(lenWords):
        s = 0
        for i in range(trainSize):
            if(emails_train[i][j] > 0 and train_labels[i] == y):
                s += 1
                
        wordProb[y][j] = s/Ycount[y]
        if(wordProb[y][j] == 0): wordProb[y][j] = 0.00001
        

# this function calculates the probability of finding a mail given it's a spam or otherwise
def P(Xi, Uy):
    ans = 1.0
    for j in range(len(Uy)):
        if Xi[j] > 0.0 :
            Xi[j] = 1
        ans *= (Uy[j]**Xi[j]) * ((1 - Uy[j])**(1-Xi[j]))
    return ans 

# running model on test data
pred_labels = [] # stores the prediction
for i in range(testSize):
    p1 = P(emails_test[i], wordProb[1]) * p
    p0 = P(emails_test[i], wordProb[0]) * (1-p)
    if(p1 > p0) :
        pred_labels.append(1)
    else:
        pred_labels.append(0)
    

# calculating the errors in the test data 
error = 0
for i in range(testSize):
    if(test_labels[i] != pred_labels[i]):
        error += 1
errorPer = 100 - (error+1 / len(test_labels) * 100)
print("error% = ", errorPer)

# function that predicts whether the given mail is spam or not
# this function return a list of all predictions, index 0 of list will correspond to the prediction for email1.txt
def checkSpamorNot():
    data = []
    import os
    import re
    os.chdir('test')


    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{file}"

            data.append(open(file_path, 'r').read())

    # for i in range(len(data)):

    len(data)
    testData = []
    pred = []
    for i in range(len(data)):
        string = data[i]
        string.lower()
        res = re.findall(r'\w+', string)
        for i in range(len(res)):
            res[i] = res[i].lower()
        vector = []
        for i in range(len(words)):
            if(words[i] in res):
                vector.append(1)
            else:
                vector.append(0)
        testData.append(vector)
    for i in range(len(testData)):
        p1 = P(testData[i], wordProb[1]) * p
        p0 = P(testData[i], wordProb[0]) * (1-p)
        if(p1 > p0) :
            pred.append(1)
        else:
            pred.append(0)
    
    return pred

print(checkSpamorNot())









