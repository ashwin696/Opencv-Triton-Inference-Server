import time
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import cv2
import sys

from nms import non_max_suppression
import numpy as np


classes_list = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic sign","fire hydrant",
"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
"hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def infer_image(img_array, client):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    outputs.append(httpclient.InferRequestedOutput("output"))
    inputs[0].set_data_from_numpy(img_array)
    try:
        response = client.infer(model_name="yolov6n", inputs=inputs)
    except InferenceServerException:
        print("Cannot connect to Triton server")
        time.sleep(5)
        sys.exit()
    return response

def decode_outputs(detects):
    bboxs = []
    conf = []
    class_ = []
    detects = detects.tolist()
    for detect in detects:
        bboxs.append(detect[:3])
        conf.append(detect[4])
        class_.append(classes_list[int(detect[5])])

    return bboxs, conf, class_

def parse_frames(video):
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # If Reading a camera, we convert to int
    try:
        device = int(video)
    except:
        device = video

    cap = cv2.VideoCapture(device)

    # Check if video opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    
    frame_cnt = 0
    
    # Read until video is completed
    while cap.isOpened():
        fps_input_stream = int(cap.get(5))
        print(fps_input_stream)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = Image.fromarray(data)
            data = np.array(data.resize((640,640), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            data = data.transpose((2, 0, 1))
            data = np.expand_dims(data, 0)
            response = infer_image(data, client)
            predictions = response.as_numpy("outputs")
            #print(predictions)
            detections = non_max_suppression(predictions, conf_thres=0.25, max_det=100)[0]
            bbox, confidence, classes = decode_outputs(detections)
            print(classes)
            frame_cnt += 1

            # For videos, add a sleep so that we read at 30 FPS
            if not isinstance(device, int):
                time.sleep(1.0 / 30)
        
        # Break the loop
        else:
            break

    print("Done reading {} frames".format(frame_cnt))

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

parse_frames("/home/jordan/Videos/intersection1.mp4")