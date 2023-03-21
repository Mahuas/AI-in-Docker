#curl -X POST -F image=@image2.jpg 'http://localhost:3600/predict' --output recieve.jpg
#docker build -t imageai:latest .
#docker run --name image-imageai -d -p 3600:3600 imageai:latest
#docker rm -f image-imageai
#docker logs image-imageai
from imageai.Detection import ObjectDetection
import os
import flask
from PIL import Image
import io
#CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

execution_path = os.getcwd()



@app.route("/predict", methods=["POST"])
def predict():
    results = ""
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            if os.path.exists("test.jpg"):
                os.remove("test.jpg")
            image.save("test.jpg")
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            if os.path.exists("output.jpg"):
                os.remove("output.jpg")
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "test.jpg"),
                                                         output_image_path=os.path.join(execution_path,
                                                                                        "output.jpg"))
            for eachObject in detections:
                results += str(eachObject["name"]) + " : " + str(eachObject["percentage_probability"]) + "     "


            data["results"] = results
            #indicate that the request was a success
            data["success"] = True
    # return the data dictionary as a JSON response
    #return flask.jsonify(data)
    return flask.send_file("output.jpg", mimetype="image/jpg", attachment_filename="output.jpg")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=3600,threaded=True)