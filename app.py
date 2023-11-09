import argparse
import io
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from flask import Flask, render_template, request, redirect, Response
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import urllib.request
from werkzeug.utils import secure_filename
import datetime

app = Flask(__name__)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

from io import BytesIO

@app.route("/detect", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return redirect(request.url)

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels

        # Count fresh and rotten items
        count_fresh = results.pred[0][:, -1].eq(0).sum().item()
        count_rotten = results.pred[0][:, -1].eq(1).sum().item()
        count_total = count_fresh + count_rotten

        # Add count labels to the image
        img_with_labels = Image.fromarray(results.ims[0])
        draw = ImageDraw.Draw(img_with_labels)
        font = ImageFont.load_default()

        # Label for fresh items in red
        draw.text((20, 10), f"Fresh: {count_fresh}", fill="red", font=font)

        # Label for rotten items in red
        draw.text((20, 30), f"Rotten: {count_rotten}", fill="pink", font=font)

        # Label for total items in blue
        draw.text((20, 50), f"Total: {count_total}", fill="blue", font=font)
    
        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        img_with_labels.save(img_savename)
        return redirect(img_savename)
    
    return render_template('detect.html')

def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=120)
            #print(results)
            #print(results.pandas().xyxy[0])
            #results.render()  # updates results.imgs with boxes and labels
            results.print()  # print results to screen
            #results.show() 
            #print(results.imgs)
            #print(type(img))
            #print(results)
            #plt.imshow(np.squeeze(results.render()))
            #print(type(img))
            #print(img.mode)
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
        
            
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                


@app.route('/cam')
def cam():
    
    return render_template('cam.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/detect2')
def detect2():
    return render_template('detect2.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regform')
def regform():
    return render_template('regform.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)