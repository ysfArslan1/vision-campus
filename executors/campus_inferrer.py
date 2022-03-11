import sys
import os
import cv2
import json
import numpy as np
from utils.config import Config
from configs.config import CFG_CAMPUS

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


class CampusInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG_CAMPUS)
        self.classes = None
        self.net = None
        self._layer_names = None
        self._output_layers = None
        self.image_blob = None
        self.layer_results = None
        self.image = None

    """self.config.path.obj_name"""

    def Load_classes(self):
        obj_path = self.config.data.path.obj
        with open(obj_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def inference_engine(self):
        cfg_path = self.config.data.path.cfg
        weight_path = self.config.data.path.weight
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._layer_names = self.net.getLayerNames()
        self._output_layers = [self._layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def proces(self, img):
        self.image_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (480, 480), swapRB=True, crop=False)
        self.net.setInput(self.image_blob, "data")
        self.layer_results = self.net.forward(self._output_layers)

    def final_prediction(self, img):

        path_txt = "weights/yolo_campus/obj .txt"
        with open(path_txt) as f:
            contents = f.readlines()
        f.close()
        img = np.asarray(img)
        img = np.float32(img)
        self.proces(img)
        height, width = img.shape[0], img.shape[1]
        boxes, confs, class_ids = [], [], []
        final_result = []
        for output in self.layer_results:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)

        merge_boxes_ids = cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.3)

        object = {
            "width": 0,
            "height": 0,
            "x": 0,
            "y": 0,
            "accuracy": 0,
            "location": ""
        }
        for i in merge_boxes_ids:
            object.update({"width": boxes[int(i)][2]})
            object.update({"height": boxes[int(i)][3]})
            object.update({"x": boxes[int(i)][0]})
            object.update({"y": boxes[int(i)][1]})
            object.update({"accuracy": confs[int(i)]})
            object.update({"location": contents[np.int64(class_ids[int(i)]).item()].rstrip('\n')})
            final_result.append(object)
        print(object)
        final_result = json.dumps(final_result)

        return {'segmentation_output': final_result}

