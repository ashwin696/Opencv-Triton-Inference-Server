import argparse
import base64
import json
import time
from collections import deque
#from concurrent.futures import as_completed
from threading import Thread
from PIL import Image
import tritonclient.http as httpclient
#from tritonclient.utils import InferenceServerException
import cv2
##import requests
from nms import non_max_suppression
import numpy as np
#from requests_futures.sessions import FuturesSession


classes_list = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic sign","fire hydrant",
"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
"hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

def read_frames(args):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # If Reading a camera, we convert to int
    try:
        device = int(args.input)
    except:
        device = args.input

    cap = cv2.VideoCapture(device)

    # Check if video opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    frame_cnt = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Encode the frame into byte data
            #data = cv2.imencode(".jpg", frame)[1]
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = Image.fromarray(data)
            data = np.array(data.resize((640,640), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            data = data.transpose((2, 0, 1))
            data = np.expand_dims(data, 0)
            queue.append(data)
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


def send_frames(payload, snd_cnt, session):

    snd_cnt += 1
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    outputs.append(httpclient.InferRequestedOutput("output"))
    inputs[0].set_data_from_numpy(payload)
    #print(json_payload)
    response = session.infer(model_name="yolov7", inputs=inputs)
    #print(response)


    return (response, snd_cnt)

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

def calculate_fps(start_time, snd_cnt):

    end_time = time.time()
    if args.client_batching:
        fps = 1.0 * args.batch_size / (end_time - start_time)
    else:
        fps = 1.0 / (end_time - start_time)

    print(
        "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
            args.batch_size, snd_cnt, fps
        )
    )
    return fps


def batch_and_send_frames(args):

    # Initialize variables
    count, exit_cnt, snd_cnt, log_cnt = 0, 0, 0, 20
    payload, futures = {}, []
    start_time = time.time()
    fps = 0
    session = httpclient.InferenceServerClient(url=api)

    while True:

        # Exit condition for the while loop. Need a better logic
        if len(queue) == 0:
            exit_cnt += 1
            # By trial and error, 1000 seemed to work best
            if exit_cnt >= 1000:
                print(
                    "Length of queue is {} , snd_cnt is {}".format(len(queue), snd_cnt)
                )
                break
        if queue:
            payload = queue.popleft()

            response, snd_cnt = send_frames(payload, snd_cnt, session)

            #futures.append(response)
            predictions = response.as_numpy("output")
            detections = non_max_suppression(predictions, conf_thres=0.25, max_det=100)[0]
            bbox, confidence, classes = decode_outputs(detections)
            print(classes)

            if snd_cnt % log_cnt == 0:
                # Calculate FPS
                fps = calculate_fps(start_time, snd_cnt)

                # Printing the response
            

                # Cleaning up futures in case futures becomes too large
                #del futures[:log_cnt]

            # Reset for next batch
            start_time = time.time()
            #payload = None

        # Sleep for 10 ms before trying to send next batch of frames
        time.sleep(args.sleep)

    # Send any remaining frames
    _, snd_cnt = send_frames(payload, snd_cnt, session)
    print(
        "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
            args.batch_size, snd_cnt, fps
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        help="Batch frames on TorchServe side for inference",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--input",
        help="Path to video file or device id",
        default="/home/ubuntu/intersection1.mp4",
    )
    parser.add_argument(
        "--client-batching",
        help="To use client side batching methodology",
        action="store_true",
    )
    parser.add_argument(
        "--sleep",
        help="Sleep between 2 subsequent requests in seconds",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    # Read frames are placed here and then processed
    queue = deque([])
    api = "localhost:8000"

    thread1 = Thread(target=read_frames, args=(args,))
    thread2 = Thread(target=batch_and_send_frames, args=(args,))
    thread1.start()
    thread2.start()