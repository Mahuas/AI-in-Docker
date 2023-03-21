# USAGE
# Start the server:
#       python app.py
# Submit a request via cURL:
#       curl -X POST -F image=@test.jpg 'http://localhost:2700/face-recognition'


from flask import Flask,jsonify,request,redirect
import face_recognition
import numpy as np
import os
from PIL import Image
import io

#CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

#加载人脸数据、姓名
known_face_encodings = []
known_face_names = []
employee_pictures = "/root/Edge/app/face-recognition/data/"
for file in os.listdir(employee_pictures):
    employee, extension = file.split(".")
    known_face_names.append(employee)
    img = face_recognition.load_image_file('/root/Edge/app/face-recognition/data/%s.jpg' % (employee))
    img_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(img_encoding)
    print(len(known_face_encodings))
    print(known_face_names)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/face-recognition', methods=['GET', 'POST'])
def upload_image():
    # 检测图片是否上传成功
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # 图片上传成功，检测图片中的人脸
            return detect_faces_in_image(file)

    # 图片上传失败，输出以下html代码
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream):
    data = {"success":False}
    # 载入用户上传的图片
    img = face_recognition.load_image_file(file_stream)
    # 为用户上传的图片中的人脸编码
    unknown_face_encoding = face_recognition.face_encodings(img)[0]

    matches = face_recognition.compare_faces(known_face_encodings,unknown_face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings,unknown_face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        print(name)
        data["success"] = True
        data["predictions"] = []
        data["predictions"].append(name)
    return jsonify(data)



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=2700,threaded=True)
