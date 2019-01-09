import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC 
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
 

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
def find_features(document):
   
    words =  set(document)
     
    features = {}
    for w in word_features:
        features[w] = (w in words)
       
    return features
    
 
featuresets = [(find_features(rev),category) for (rev,category) in doc1]
training_set = featuresets
#print(training_set)
testing_set = featuresets   
#print (featuresets)
#classifier = nltk.NaiveBayesClassifier.train(training_set)
#print("Original Naive algo accuracy percent:" ,(nltk.classify.accuracy(classifier,testing_set))*100)
#classifier.show_most_informative_features(15)
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
#print("MNB Classifier algo accuracy percent:" ,(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
name = 'banana'

print(MNB_classifier.classify(find_features([name])))
 

 
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print(BernoulliNB_classifier.classify(find_features([name])))


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print(LogisticRegression_classifier.classify(find_features([name])))


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print(SGDClassifier_classifier.classify(find_features([name])))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
print(LinearSVC_classifier.classify(find_features([name])))


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
print(NuSVC_classifier.classify(find_features(['what is orange'])))