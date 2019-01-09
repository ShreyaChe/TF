import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
 
#documents = [(list(movie_reviews.words(fileid)), category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]
 

#print(documents[1])
#random.shuffle(documents)
lines = [line.rstrip('\n') for line in open('C:\\Users\\shrey_vr\\Desktop\\Datascience\\tut\\oldscatterd.txt')]
doc1 = []
all_words =[]

for x in lines:
        if x:
            class_text,  sent_text= x.split("==")
            word_tokens = word_tokenize(sent_text)
            wrd = []
            for w in word_tokens:
                all_words.append(w.lower())
                wrd.append(w.lower())
            doc = []
            doc = [wrd,class_text]
            doc1.append(doc)
             

#print(doc1)
random.shuffle(doc1)
#print(lines[1])
#print(documents[1])
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())
#print(word_features)
def find_features(document):
    words =  set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    

featuresets = [(find_features(rev),category) for (rev,category) in doc1]
print (featuresets )