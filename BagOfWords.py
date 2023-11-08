import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datautils import review_to_words
from sklearn.ensemble import RandomForestClassifier

train = None
test = None
vectorizer = None
train_data_features = None
test_data_features = None
clean_train_reviews = None

def loadData():
    global train,test
    train = pd.read_csv("DATA\labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
    test = pd.read_csv(r"DATA\testData.tsv", header=0,delimiter="\t", quoting=3)
    # unlabeled = pd.read_csv(r"DATA\unlabeledTrainData.tsv", header=0,delimiter="\t", quoting=3)
    
def processTrain():
    global clean_train_reviews
    num_reviews = train["review"].size
    clean_train_reviews = []
    for i in range(num_reviews):
        clean_train_reviews.append(review_to_words(train["review"][i]))
        if i%1000==0:
            print("Review ",(i+1)," of ",num_reviews)

def processTest():
    global clean_test_reviews
    num_reviews = test["review"].size
    clean_test_reviews = []
    for i in range(num_reviews):
        clean_test_reviews.append(review_to_words(test["review"][i]))
        if i%1000==0:
            print("Review ",(i+1)," of ",num_reviews)
     

def vectorizeWords(max_features=5000):
    global train_data_features,vectorizer,test_data_features
    vectorizer = CountVectorizer(analyzer = "word",
                                max_features = max_features)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()    
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

def getVocabStats(n=10):
    vocab = vectorizer.get_feature_names_out()
    dist = np.sum(train_data_features, axis=0)
    vocabdist = zip(vocab,dist)
    vocabdist = sorted(vocabdist, key=lambda x: x[1], reverse=True) 
    print("Top ",n," most frequently used words:")
    for i in range(n):
        print(vocabdist[i])
 
def randForest(fileName="Bag_of_Words_model.csv",n=100):
    forest = RandomForestClassifier(n_estimators = n) 
    forest = forest.fit(train_data_features, train["sentiment"])
    result = forest.predict(test_data_features)
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    # Use pandas to write the comma-separated output file
    output.to_csv(fileName, index=False, quoting=3)

if __name__=='__main__':
    loadData()
    processTrain()
    processTest()
    vectorizeWords(max_features=10000)
    randForest(fileName="BoW10000Feats.csv")
    randForest(fileName="BoW10Kwith200Est.csv",n=200)
    randForest(fileName="BoW10Kwith50Est.csv",n=50)


