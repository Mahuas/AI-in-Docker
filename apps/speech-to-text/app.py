#curl -X POST -F audio=@asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac 'http://localhost:2900/speechtotext'
#curl -X POST -F audio=@asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0005.flac 'http://localhost:2900/speechtotext'

#docker rm -f image-speech
#docker run --name image-speech -d -p 2900:2900 speechtotext:latest
#docker build -t speechtotext:latest .
#docker logs image-speech
#ValueError: Tensor("cond_1/pred_id:0", shape=(), dtype=bool) must be from the same graph as Tensor("front/conv_in/moments/normalize/mean:0", shape=(128,), dtype=float32).
# -*- coding: utf-8 -*-


import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
import flask
import json
import os


#CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
graph = tf.get_default_graph()

app = flask.Flask(__name__)

# set log level to debug
tf.sg_verbosity(10)
batch_size = 1  # batch size
# vocabulary size
voca_size = data.voca_size

x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))
# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)
# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len,
                                           merge_repeated=False)

def load_model():
    global y
    # to dense tensor
    y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
    global graph
    graph = tf.get_default_graph()

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_


@app.route("/speechtotext",methods=["POST"])
def speechtotext():
    resultdata = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("audio"):
            global y, graph
            resultstr = ""
            audio = flask.request.files["audio"]
            if os.path.exists("asset/data/test.flc"):
                os.remove("asset/data/test.flc")
            audio.save("asset/data/test.flc")
            y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
            # load wave file
            wav, _ = librosa.load('asset/data/test.flc', mono=True, sr=16000)
            # get mfcc feature
            mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])
            # run network
            with graph.as_default():
                # init variables
                tf.sg_init(sess)
                # restore parameters
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
                # run session
                label = sess.run(y, feed_dict={x: mfcc})
                for index_list in label:
                    resultstr += index2str(index_list)
                # print label
                resultdata["result"] = resultstr
                resultdata["success"] = True

    return flask.jsonify(resultdata)



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=2900,threaded=True)