import os
from pathlib import Path
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2

def load_person_detection_retail_0013_to_IE(PI=False):
    cpu_extension = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    model_xml = "/".join(os.path.abspath(__file__).split("/")[
                         :-1]) + "/person-detection-retail-0013/CHANGE/person-detection-retail-0013.xml"
    if PI:
        model_xml=model_xml.replace("/CHANGE/","FP16")
    else:
        model_xml=model_xml.replace("/CHANGE/","FP32")

    plugin = IECore()
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    if cpu_extension:
        plugin.add_extension(cpu_extension, "CPU")
    supported_layers = plugin.query_network(network=net, device_name="CPU")
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        exit(1)
    exec_net = plugin.load_network(net, "CPU")
    input_blob = next(iter(net.inputs))
    input_shape = net.inputs[input_blob].shape
    print('loaded completly')
    return exec_net, input_shape

def preprocessing(input_image, height, width):
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)
    return image

def nms(dets, score_threshold=0.3, beta=3):
    """
    Las detecciones deben venir en formato:
    [x1,y1,x2,y2,acc,class_to_be_detected]
    """
    def iou(bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    filtered_dets = []
    total_dets = sorted(dets.copy(), key=lambda x: x[4], reverse=True)
    while len(total_dets) > 0:
        for i in range(1, len(total_dets)):
            IOU = iou(total_dets[i][:4], total_dets[0][:4])
            total_dets[i][4] *= np.exp(-beta * IOU) #(1-IOU)
        temp = []
        for i in total_dets:
            if i[4] >= score_threshold:
                temp.append(i)
        if len(temp) > 0:
            filtered_dets.append(temp[0])
        total_dets = sorted(temp[1:].copy(), key=lambda x: x[4], reverse=True)
        del temp
    return filtered_dets

def get_results_from_person_detection_retail_0013(exec_net, input_shape ,image, dets_confidence=0.4,nms_threshold=0.3,nms_bets=3):
    def perform_inference(exec_net, image, h, w):
        preprocessed_image = preprocessing(image, h, w)
        input_blob = next(iter(exec_net.inputs))
        output = exec_net.infer({input_blob: preprocessed_image})
        return output

    def filter(result, h, w):
        dets = result['detection_out'][0][0]
        dets_fil = []
        for det in dets:
            if det[0] == -1:
                break
            else:
                if det[1] in [1,2,3,4] and float(det[2]) >= dets_confidence:    #[1-person,2-bicycle,3-car,4-motorbike]
                    x1, y1, x2, y2 = det[3] * w, det[4] * h, det[5] * w, det[6] * h
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil
    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)

