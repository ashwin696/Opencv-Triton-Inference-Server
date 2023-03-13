import torch
import onnxruntime as ort
import tritonclient.http as httpclient


class Inferer:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def torch_ifner(self):
        return

    def onnx_infer(self, input_img, cuda=False):
        '''
        Function to inference with onnxruntime
        '''
        # if cuda:
        #     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # else:
        #     providers = ['CPUExecutionProvider']
        # providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider']) # creating a session for inference
        inputs = [(i.name) for i in session.get_inputs()] [0] # getting the input layer name
        outputs = [(i.name) for i in session.get_outputs()]# getting the output layer name
        out = session.run(outputs,{inputs:input_img})
        return out

    def openvino_ifner(self):
        return

    def triton_infer(self, api, input_img):
        '''
        Function to inference with triton inference server 
        using triton client
        '''
        session = httpclient.InferenceServerClient(url=api)
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('images', [1, 3, 640, 640], "FP32"))
        outputs.append(httpclient.InferRequestedOutput("output"))
        inputs[0].set_data_from_numpy(input_img)
        response = session.infer(model_name="yolov7", inputs=inputs)
        return response