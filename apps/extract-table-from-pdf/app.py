#curl -X POST -F file=@foo.pdf 'http://127.0.0.1:3400/extract'
#curl -X POST -F file=@xialingying.pdf 'http://127.0.0.1:3400/extract'
#docker build -t extract-table:latest .
#docker run --name image-extracttable -d -p 3400:3400 extract-table:latest
#docker rm -f image-extracttable
#docker logs image-extracttable
#docker rmi -f

import flask
import os
import camelot
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False



@app.route("/extract", methods=["post"])
def extract():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("file"):
            pdf = flask.request.files["file"]
            if pdf:
                print("pdf has recieved")
            if os.path.exists("/home/test.pdf"):
                os.remove("/home/test.pdf")
            pdf.save("/home/test.pdf")
            #print(pdf)
            print("test.pdf")
            tables = camelot.read_pdf("test.pdf")
            data["extract_report"] = tables[0].parsing_report
            data["results"] = tables[0].df.to_json(orient='records')
            data["success"] = True
            csvfile = tables[0].to_csv('test.csv')
            response = flask.make_response(flask.send_from_directory("/home/",filename="test.csv",attachment=True))
            #response.headers["Content-Disposition"] = "attachment; filename=test.csv"
            #response.headers["Content-type"] = "application/octet-stream"

    #return response
    return flask.send_file("test.csv",mimetype="text/csv",attachment_filename="test.csv")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3400,threaded=True)
