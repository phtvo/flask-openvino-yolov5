import os, sys
from time import time 
import random
from numba.cuda.simulator.api import detect 
import numpy as np 
import cv2
from openvino.inference_engine import IENetwork, IECore
import ngraph as ng

from .utils import pad_img, YoloParams, parse_yolo_detection, nms, letterbox, parse_scale
class yolo:

    def __init__(self, path2xml,  nclasses,params:dict, classes=None, device='CPU', 
                cpu_extension=None, conf_thresh=0.5, iou_thresh=0.5) -> None:
        self.nclasses = nclasses
        ie = IECore()

        model_bin = os.path.splitext(path2xml)[0] + ".bin"
        if cpu_extension and 'CPU' in device:
            ie.add_extension(cpu_extension, "CPU")
        self.net =IECore().read_network(model=path2xml, weights=model_bin)
        function = ng.function_from_cnn(self.net)

        self.input_blob = next(iter(self.net.input_info))

        #  Defaulf batch_size is 1
        self.net.batch_size = 1
        self.batch, self.ch, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

        self.exec_net = ie.load_network(network=self.net, num_requests=2, device_name=device)
        self.cur_request_id = 0
        self.params = YoloParams(params)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.classes = classes 
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(nclasses)]


    def preprocess(self, img:np.ndarray) -> np.ndarray:
        """ RGB -> (bs, ch, w, h)
        Args:
            img (np.ndarray): RGB image

        Returns:
            np.ndarray: (1, 3, w, h) 
        """
        in_frame = letterbox(img, (self.w, self.h), color=(0,0,0))
        #cv2.imwrite('frame.jpg', in_frame)
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return in_frame.reshape((self.batch, self.ch, self.h, self.w))

    def run(self, image, request_id=1):
        org_h, org_w = image.shape[:2]
        image = self.preprocess(image)
        re_w, re_h = self.w, self.h
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: image})
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            result = []
            output = self.exec_net.requests[self.cur_request_id].output_blobs
            for layer_name, out_blob in output.items():
                #print(self.net.outputs[layer_name].shape)
                out_blob = out_blob.buffer.reshape(self.net.outputs[layer_name].shape)
                coords = parse_yolo_detection(out_blob, (re_w, re_h),(org_w, org_h), 
                            self.params.anchors[parse_scale(re_w, out_blob.shape[2])], self.params.num_anchors, self.conf_thresh)
                result.append(coords) 
            result = np.concatenate(result)
            detected_result = nms(predictions= result, nclasses= self.nclasses, conf_thresh=None, iou_thresh=self.iou_thresh)
            return detected_result
        else:
            return None 

    def detect_in_image(self, src:np.ndarray, saveimg=None):
        detections = self.run(src, 0)
        start_time = time()
        if isinstance(detections, np.ndarray):
            for i in range(detections.shape[0]):
                det = detections[i]
                x1,y1,x2,y2 = [int(each) for each in det[:4] ]
                conf = det[4]
                clss = int(det[-1])
                det_label = str(clss) if not self.classes else self.classes[clss]
                cv2.rectangle(src, (x1, y1), (x2, y2), self.colors[clss], 1)
                cv2.putText(src,
                            "#" + det_label + ' ' + str(round(conf* 100, 1)) + ' %',
                            (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, self.colors[clss], 1)
            if saveimg:
                cv2.imwrite(saveimg, src)
            print(f'Total processed time = {round((time() - start_time) * 1000, 2)}ms')
        return src

    def detect_api(self, src:np.ndarray):
        """Run single detection

        Args:
            src (np.ndarray): image
        Returns:
            dict: ( result:dict('label':list(list(x1:int, y1:int, x2:int, y2:int))))
        """
        detections = self.run(src, 0)
        start_time = time()
        result = {}
        if isinstance(detections, np.ndarray):
            for i in range(detections.shape[0]):
                det = detections[i]
                x1,y1,x2,y2 = [int(each) for each in det[:4] ]
                conf = det[4]
                clss = int(det[-1])
                det_label = str(clss) if not self.classes else self.classes[clss]

                if det_label in result:
                    result[det_label].append((x1, y1, x2, y2))
                else:
                    result.update({det_label : [ (x1, y1, x2, y2) ]})
            #print(f'Total processed time = {round((time() - start_time) * 1000, 2)}ms')
        return result
            


    def detect_in_videos(self, src):
        is_async_mode = True
        cap = cv2.VideoCapture(src)
        number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
        wait_key_code = 1
        # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
        if number_input_frames != 1:
            ret, frame = cap.read()
        else:
            is_async_mode = False
            wait_key_code = 0
        is_async_mode = 1
        cur_request_id = 0
        next_request_id = 1
        render_time = 0
        parsing_time = 0
        while cap.isOpened():
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
            if is_async_mode:
                ret, next_frame = cap.read()
                request_id = next_request_id
            else:
                ret, frame = cap.read()
                request_id = cur_request_id

            if not ret:
                break

            inframe = frame
            # Start inference
            start_time = time()
            detections = self.run(inframe, request_id=request_id)
            det_time = time() - start_time 
            start_time = time()
            if isinstance(detections, np.ndarray):
                for i in range(detections.shape[0]):
                    det = detections[i]
                    x1,y1,x2,y2 = [int(each) for each in det[:4] ]
                    conf = det[4]
                    clss = int(det[-1])
                    det_label = str(clss) if not self.classes else self.classes[clss]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors[clss], 1)
                    cv2.putText(frame,
                                "#" + det_label + ' ' + str(round(conf* 100, 1)) + ' %',
                                (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, self.colors[clss], 1)

            # Draw performance stats over frame
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1e3)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)
            parsing_message = "YOLO parsing time is {:.3f} ms".format(det_time * 1e3)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(inframe.shape[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 200), 1)
            #start_time = time()
            render_time = time() - start_time
            
            cv2.imshow("DetectionResults", frame)
            key = cv2.waitKey(wait_key_code)
            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame
        
            # ESC key
            if key == 27:
                break
            # Tab key
            if key == 9:
                self.exec_net.requests[cur_request_id].wait()
                is_async_mode = not is_async_mode
                #log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

        cv2.destroyAllWindows()