import os
import argparse
import cv2
import numpy as np
import time
from openvino.inference_engine import IECore, IENetwork

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def get_args():
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()
    return args


def load_to_IE(model_xml, cpu_extension):
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


def nms(dets, score_treshold=0.3, beta=3):
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
            total_dets[i][4] *= np.exp(-beta * IOU)
        temp = []
        for i in total_dets:
            if i[4] >= score_treshold:
                temp.append(i)
        if len(temp) > 0:
            filtered_dets.append(temp[0])
        total_dets = sorted(temp[1:].copy(), key=lambda x: x[4], reverse=True)
        del temp
    return filtered_dets


def put_in_frame(filtered_dets, image):
    for det in filtered_dets:
        x1, y1, x2, y2 = det[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return image


def perform_on_image(args, function):
    image_path = '/home/efmb/Desktop/openvino_models_testing/image2.jpg'
    image = cv2.imread(image_path)

    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)

    filtered_dets = function(exec_net, image, input_shape)

    print(filtered_dets)
    print(len(filtered_dets))

    result_image = put_in_frame(filtered_dets, image)

    cv2.imshow('win', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perform_on_video(args, function):
    video = "/home/efmb/Desktop/openvino_models_testing/video.avi"
    cam = cv2.VideoCapture(video)
    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)
    while True:
        _, image = cam.read()
        if not _:
            break
        filtered_dets = function(exec_net, image, input_shape)
        result_image = put_in_frame(filtered_dets, image)
        cv2.imshow('win', result_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def get_results_from_ssd_mobilenet_v2_coco(exec_net, image, input_shape):
    def perform_inference(exec_net, image, h, w):
        preprocessed_image = preprocessing(image, h, w)
        input_blob = next(iter(exec_net.inputs))
        output = exec_net.infer({input_blob: preprocessed_image})
        return output

    def filter(result, h, w):
        dets = result['DetectionOutput'][0][0]
        dets_fil = []
        for det in dets:
            if det[0] == -1:
                break
            else:
                if det[1] == 1 and float(det[2]) >= 0.4:
                    x1, y1, x2, y2 = det[3] * w, det[4] * h, det[5] * w, det[6] * h
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)


def get_results_from_pedestrian_detection_adas_0002(exec_net, image, input_shape):
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
                if det[1] == 1 and float(det[2]) >= 0.4:
                    x1, y1, x2, y2 = det[3] * w, det[4] * h, det[5] * w, det[6] * h
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)


def get_results_from_person_detection_retail_0002(exec_net, image, input_shape):
    def perform_inference(exec_net, image, h, w):
        preprocessed_image = preprocessing(image, h, w)
        # input_blob = next(iter(exec_net.inputs))
        output = exec_net.infer({'data': preprocessed_image,
                                 'im_info': [544, 992, 992 / w, 544 / h, 992 / w, 544 / h]
                                 })
        return output

    def filter(result, h, w):
        dets = result['detection_out'][0][0]
        dets_fil = []
        for det in dets:
            if int(det[0]) == -1:
                break
            if det[1] == 1 and float(det[2]) >= 0.4:
                dets_fil.append([det[3] * w, det[4] * h, det[5] * w, det[6] * h, round(float(det[2]), 5), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)


def get_results_from_person_detection_retail_0013(exec_net, image, input_shape):
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
                if det[1] == 1 and float(det[2]) >= 0.4:
                    x1, y1, x2, y2 = det[3] * w, det[4] * h, det[5] * w, det[6] * h
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)


def get_results_from_person_vehicle_bike_detection_crossroad_1016(exec_net, image, input_shape):
    def perform_inference(exec_net, image, h, w):
        preprocessed_image = preprocessing(image, h, w)
        input_blob = next(iter(exec_net.inputs))
        output = exec_net.infer({input_blob: preprocessed_image})
        return output

    def filter(result, h, w):
        dets = result['653'][0][0]
        dets_fil = []
        for det in dets:
            if det[0] == -1:
                break
            else:
                if det[1] == 1 and float(det[2]) >= 0.04:
                    x1, y1, x2, y2 = det[3] * w, det[4] * h, det[5] * w, det[6] * h
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    h, w = input_shape[2:]
    img_h, img_w = image.shape[:2]
    return filter(perform_inference(exec_net, image, h, w), img_h, img_w)


def main():
    args = get_args()
    #
    # Fallas como full_screen_box - bueno para ciertas circunstacias por medir aun.
    # perform_on_image(args,get_results_from_ssd_mobilenet_v2_coco)
    # perform_on_video(args,get_results_from_ssd_mobilenet_v2_coco)
    #
    # Detecciones muy erraticas
    # perform_on_image(args,get_results_from_pedestrian_detection_adas_0002)
    # perform_on_video(args,get_results_from_pedestrian_detection_adas_0002)
    #
    # Muy lento
    # perform_on_image(args,get_results_from_person_detection_retail_0002)
    # perform_on_video(args,get_results_from_person_detection_retail_0002)
    #
    # el mejor - visualmente en detecciones
    perform_on_image(args, get_results_from_person_detection_retail_0013)
    # perform_on_video(args,get_results_from_person_detection_retail_0013)
    #
    # El siguiente es para ver desde lejos, cercano no funca
    # perform_on_image(args,get_results_from_person_vehicle_bike_detection_crossroad_1016)
    # perform_on_video(args,get_results_from_person_vehicle_bike_detection_crossroad_1016)


if __name__ == "__main__":
    main()