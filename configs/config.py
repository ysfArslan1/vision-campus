# -- coding: utf-8 --
"""Model config in json format"""

CFG_CAMPUS = {
    "data": {
        "path": {
            "weight":"/opt/project/weights/yolo_campus/yolov3_final.weights",
            "cfg":"/opt/project/weights/yolo_campus/yolov3.cfg",
            "obj":"/opt/project/weights/yolo_campus/obj.txt"
        },
        "image_size": None,
        "load_with_info": None
    },
    "train": {
        "batch_size": None,
        "buffer_size": None,
        "epoches": None,
        "val_subsplits": None,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": None,
        "up_stack": {
            "layer_1": None,
            "layer_2": None,
            "layer_3": None,
            "layer_4": None,
            "kernels": None
        },
        "output": None
    }
}