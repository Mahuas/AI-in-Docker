#curl -X POST -F image=@dataset/one/covid/Chest.jpeg 'http://localhost:3800/predict' --output output.jpg
#docker build -t covid-19:latest .
#docker run --name image-covid -d -p 3800:3800 covid-19:latest
#docker logs image-covid
#docker rm -f image-covid


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from keras.optimizers import SGD
from Model.IncludeNet import IncludeNet
import numpy as np
import cv2
import flask
import os
import PIL
from PIL import Image
import io

#CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

classLabels = ["covid", "normal", "vira neumonia"]



@app.route("/predict", methods=["POST"])
def predict():
    data = dict()
    data["success"] = False
    output_dir = "/home/result/"
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # 获取图片
            image = flask.request.files["image"]
            if os.path.exists("input.jpg"):
                os.remove("input.jpg")
            image.save("input.jpg")

            image_path = "input.jpg"
            size = 50
            sp = SimplePreprocessor(size, size)
            iap = ImageToArrayPreprocessor()
            sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
            (data, labels) = sdl.single_load(image_path)
            data = data.astype("float") / 255.0
            model = load_model('./SavedModel/amin.hdf5')
            preds = model.predict(data, batch_size=size).argmax(axis=1)
            image = cv2.imread(image_path)
            # image=ReadyToUseImage(image)
            cv2.putText(image, "Label: {}".format(classLabels[preds[preds[0]]]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if os.path.exists("/home/result/output.jpg"):
                os.remove("/home/result/output.jpg")
            image.save("/home/result/output.jpg")
    if os.path.exists("/home/result/output.jpg"):
        return flask.send_file("/home/result/output.jpg", mimetype="image/jpg")
    else:
        data["error"] = "no such a file named '/home/result/output.jpg'!"
        return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3800,threaded=True)