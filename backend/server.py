from flask import Flask, redirect, request, url_for
import nltk
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle
import json

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "localhost:3000"
    }
})
model = pickle.load(open("model.pkl", 'rb'))

lem = WordNetLemmatizer()
stem = PorterStemmer()


def transform_text(text):
    text.lower()
    words = nltk.word_tokenize(text)
    y = []
    for i in words:
        if (i.isalnum() and i not in stopwords.words('english') and i.isnumeric() != True):
            x = stem.stem(i)
            y.append(x)
    return ' '.join(y)


vector = pickle.load(open("vectorized.pkl", 'rb'))


@app.route("/", methods=['GET', 'POST'])
def helloWorld():
    if(request.method == 'POST'):
        # print(request.data,"hegfjheifdwmnvcbsyhdgbfwjnedfyhwgbefjgwyifebn")
        a = request.data
        a = json.loads(a.decode('utf-8'))
        a = transform_text(a['text'])
        print(a)
        x = vector.transform([a])
        print(x)
        result = model.predict(x)
        if result[0] == 1:
            return "Spam"
        return "Not Spam"
    else:
        return 'get'


if __name__ == "__main__":
    app.run(debug=True)
