# coding: utf-8
# curl -X POST -F image=@test1.jpg 'http://localhost:3500/predict'
# docker build -t verify-code:latest .
# docker run --name image-verify -d -p 3500:3500 verify-code:latest
# docker rm -f image-verify
# docker logs image-verify
# docker tag 85ba89073fea registry.cn-hangzhou.aliyuncs.com/k3s_ssh/verify-code:latest

import cv2
import numpy as np
from keras import models
import io
import pretreatment
from mlearn_for_image import preprocess_input
import flask
import os
from PIL import Image

#CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def get_text(img, offset=0):
    text = pretreatment.get_text(img, offset)
    text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    text = text / 255.0
    h, w = text.shape
    text.shape = (1, h, w, 1)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    results = ""
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            if os.path.exists("test.jpg"):
                os.remove("test.jpg")
            image.save("test.jpg")

            # 读取并预处理验证码
            img = cv2.imread("test.jpg")
            text = get_text(img)
            imgs = np.array(list(pretreatment._get_imgs(img)))
            imgs = preprocess_input(imgs)

            # 识别文字
            model = models.load_model('model.h5')
            label = model.predict(text)
            label = label.argmax()
            fp = open('texts.txt', encoding='utf-8')
            texts = [text.rstrip('\n') for text in fp]
            text = texts[label]
            print(text)
            # 获取下一个词
            # 根据第一个词的长度来定位第二个词的位置
            if len(text) == 1:
                offset = 27
            elif len(text) == 2:
                offset = 47
            else:
                offset = 60
            text = get_text(img, offset=offset)
            if text.mean() < 0.95:
                label = model.predict(text)
                label = label.argmax()
                text = texts[label]
                print(text)

            # 加载图片分类器
            model = models.load_model('12306.image.model.h5')
            labels = model.predict(imgs)
            labels = labels.argmax(axis=1)
            for pos, label in enumerate(labels):
                results += str(pos // 4)+str(pos % 4)+str(texts[label])


            data["predictions"] = results

            # indicate that the request was a success
            data["success"] = True
    return flask.jsonify(data)




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3500,threaded=True)
