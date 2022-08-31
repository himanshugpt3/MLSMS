from flask import Flask, jsonify
import pickle
import re
import string
import pickle
import numpy as np
from os.path import  join,dirname
app = Flask(__name__)

def text_preprocessing(content):

    text = re.sub("[^a-zA-Z0-9]"," ",content)
    content=text
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    punc = [token for token in content if token not in string.punctuation]
    punc = ''.join(punc)

    return ' '.join([word for word in punc.split() if word .lower() not in stop_words])    

@app.route("/predict/<string:content>")
def main(content):

    result={
            "content" : content,
            "details" : {
                "safe" : True
            }
        }
    kaggle_pikle=join(dirname(__file__),"finals_model.pkl")
    filename2=join(dirname(__file__),"final_cv.pkl")
    xforce_pikle=join(dirname(__file__),"finals_model2.pkl")
    filename4=join(dirname(__file__),"final_cv2.pkl")


    kaggle_trained_model = pickle.load(open(kaggle_pikle, 'rb'))
    tfidf = pickle.load(open(filename2, 'rb'))

    xforce_trained_model=pickle.load(open(xforce_pikle,'rb'))
    tfidf2=pickle.load(open(filename4,'rb'))

    test=[]
    test.append(text_preprocessing(content))

    a=tfidf.transform(test).toarray()
    a=np.array(a)

    b=tfidf2.transform(test).toarray()
    b=np.array(b)

    kaggle_prediction=kaggle_trained_model.predict(a)
    xforce_prediction=xforce_trained_model.predict(b)

    if(kaggle_prediction[0]==1 or xforce_prediction[0]==1):
        result={
            "content" : content,
            "details" : {
                "safe" : False
            }
        }
    return jsonify(result)
