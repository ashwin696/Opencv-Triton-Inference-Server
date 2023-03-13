import re
import cv2 as cv
import numpy as np
from PIL import Image

presets = dict(
    empty = 'synth:',
    lena = 'synth:bg=lena.jpg:noise=0.1',
    chess = 'synth:class=chess:bg=lena.jpg:noise=0.1:size=640x480',
    book = 'synth:class=book:bg=graf1.png:noise=0.1:size=640x480',
    cube = 'synth:class=cube:bg=pca_test1.jpg:noise=0.0:size=640x480'
)

def create_capture(source = 0, fallback = presets['chess']):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    This function is taken from opencv repo:
    https://github.com/opencv/opencv/blob/0052d46b8e33c7bfe0e1450e4bff28b88f455570/samples/python/video.py#L167 

    '''
    source = str(source).strip()

    # Win32: handle drive letter ('c:', ...)
    source = re.sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [re.sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv.VideoCapture(source)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap


def pre_process_img(frame, size=(640,640)):
    '''
    this function converts cv2 image array into suitable for yolov6 model
    inference. 
    (BGR->RGB->Resize(640, 640)->Reshape)
    '''
    data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    data = Image.fromarray(data) # converting to Pillow image for resizing
    data = np.array(data.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    data = data.transpose((2, 0, 1)) # shifting (W, H, C) -> (C, W, H)
    data = np.expand_dims(data, 0) # increasing dimension to (1, C, W, H)

    return data
