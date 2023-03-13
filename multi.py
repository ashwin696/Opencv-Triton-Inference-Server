
import time
import cv2
import numpy as np
from nms import non_max_suppression
from multiprocessing.pool import ThreadPool
from collections import deque
from utils import create_capture, pre_process_img
from infer import Inferer

def main(file, model):

    start = time.time()
    with open("./labels.txt", "r") as fp:
        classes_list = fp.readlines()

    cap = create_capture(file)
    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()
    ori_frame = None
    count = 0
    inferer = Inferer(model)
    while True:
        while len(pending) > 0 and pending[0].ready():
            res = pending.popleft().get()
            out = inferer.onnx_infer(res)
            detects = non_max_suppression(out[0])[0]
            detects = detects.tolist()
            for detect in detects:
                bbox = list(map(lambda a: int(a), detect[:4]))
                conf = detect[4]
                class_ = classes_list[int(detect[5])].strip() # to remove \n 
                cv2.rectangle(ori_frame, bbox[:2],bbox[2:],[225, 255, 255],5)
            cv2.imwrite("image-2.png", ori_frame)
        if len(pending) < threadn:
            
            _ret, frame = cap.read()
            ori_frame = frame
            if _ret:
                task = pool.apply_async(pre_process_img, (frame.copy(),))
                pending.append(task)
                count += 1
            else:
                break
    print(f"Done reading {count} frames, time: {time.time()-st} ")


if __name__ == '__main__':
    
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
        default="/home/jordan/Videos/intersection1.mp4",
    )
    args = parser.parse_args()
    file = "/home/jordan/Videos/min.mp4"
    main(file, "./model.onnx")