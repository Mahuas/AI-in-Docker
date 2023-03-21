#curl -d '{"weight": "politics"}' -H 'Content-Type: application/json' -X POST 'http://127.0.0.1:3100/textGen'
#curl -d '{"weight": "relationship"}' -H 'Content-Type: application/json' -X POST 'http://127.0.0.1:3100/textGen'
#curl -d '{"weight": "hacknews/cellponeOS"}' -H 'Content-Type: application/json' -X POST 'http://127.0.0.1:3100/textGen'
#docker run --name container-text-genrnn -d -p 3100:3100 text-genrnn:latest

import os
import sys
from textgenrnn import textgenrnn
import flask
import numpy

#CPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
data = dict()
def generate_samples(self, n=1, temperatures=[0.2, 0.5, 1.0], **kwargs):
    for temperature in temperatures:
        print('#'*20 + '\nTemperature: {}\n'.format(temperature) + '#'*20)
        key = 'Temperature: {}'.format(temperature)
        text = self.generate(n, temperature=temperature, **kwargs, return_as_list=True)
        data[key] = text
        print(type(text))
        print(text)

@app.route("/textGen", methods=["post"])
def textGen():
    data["success"] = False
    if flask.request.method =="POST":
        jsonstr = flask.request.get_json()
        print(jsonstr)
        weight = jsonstr['weight']
        data["success"] = True
        if weight == 'hacknews':
            textgen = textgenrnn('weights/hacker_news.hdf5')
            generate_samples(textgen)
        elif weight == 'relationship':
            textgen = textgenrnn('weights/reddit_legaladvice_relationshipadvice.hdf5')
            generate_samples(textgen)
        elif weight == 'cellphoneOS':
            textgen = textgenrnn('weights/reddit_apple_android.hdf5')
            generate_samples(textgen)
        elif weight == 'politics':
            textgen = textgenrnn('weights/reddit_rarepuppers_politics.hdf5')
            generate_samples(textgen)
    return flask.jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=3100,threaded=True)
