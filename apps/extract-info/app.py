#curl -X POST -F text=@test.txt 'http://127.0.0.1:2600/extract


import flask
import os
from cocoNLP.extractor import extractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def extract_info(text):
    ex = extractor()
    results = dict()
    #extract email
    results['emails'] = ex.extract_email(text)
    #extract phone number
    results['cellphones'] = ex.extract_cellphone(text,nation='CHN')
    #extract names
    results['times'] = ex.extract_time(text)
    #extract locations
    results['locations'] = ex.extract_locations(text)
    return results


@app.route("/extract", methods=["post"])
def extract():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("text"):
            word = flask.request.files["text"].read().decode("utf-8")
            print(word)
            results = extract_info(word)
            data["results"] = []
            data["results"].append(results)

            data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2600,threaded=True)
