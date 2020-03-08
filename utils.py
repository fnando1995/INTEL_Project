import argparse

from network.utils import *
from tracker.tracker import Tracker

def get_args():
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    base_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    pi_desc = "boolean for load neural network in format FP16 and use device MYRIAD or format FP32 ans use CPU- default FALSE"
    video_desc = "fullpath for video to be used - default '$PATH/data/videos/video.avi'"
    save_file_desc = "fullpath for text file to save countings - default '$PATH/data/texts/save.txt'"
    figure_file_desc = "fullpath for figure file .npy to generate regions - default '$PATH/figures/view1.npy'"
    parser.add_argument("-pi",
                        type=bool,
                        default=False,
                        help=pi_desc)
    parser.add_argument("-video_filepath",
                        type=str,
                        default= base_dir+ "/data/videos/video.avi",
                        help=video_desc)
    parser.add_argument("-save_filepath",
                        type=str,
                        default= base_dir + "/data/texts/save.txt",
                        help=save_file_desc)
    parser.add_argument("-figure_filepath",
                        type=str,
                        default= base_dir + "/figures/view1.npy",
                        help=figure_file_desc)
    args = parser.parse_args()
    return args

def put_in_frame(filtered_dets, image):
    for det in filtered_dets:
        x1, y1, x2, y2 = det[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return image

def put_tracked_in_frame(tracked,frame):
    for trk in tracked:
        bbox = trk.get_state()[0]
        id = str(trk.get_id())
        startX, startY,endX, endY = np.array(bbox[:4]).astype(int)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.putText(frame,id,(startX,startY),0,1,(0,255,255),3)
    return frame

def run_project_demo(args):
    # Load video
    video_path = args.video_filepath
    video = cv2.VideoCapture(video_path)
    # Load network (for CPU or PI/NCS2)
    pi_usage = args.pi
    net,net_input_shape = load_person_detection_retail_0013_to_IE(pi_usage)
    # Load Tracker
    figure_path = args.figure_filepath
    tracker = Tracker(figure_path)
    # Loop in video
    FLAG = True
    while FLAG:
        _,frame = video.read()
        if not _:
            print("fin de video")
            FLAG = False
            continue
        filtered_detections = get_results_from_person_detection_retail_0013(net,net_input_shape,frame)
        # frame_with_filtered_detections = put_in_frame(filtered_detections,frame)
        # cv2.imshow("detecting", frame_with_filtered_detections)
        # tracked_detections  = tracker.track_dets(filtered_detections)
        # frame_with_tracked_detections = put_tracked_in_frame(tracked_detections,frame)
        # cv2.imshow("tracking", frame_with_tracked_detections)
        tracker.track_dets(filtered_detections)
        frame_with_tracking_regions_and_counting = tracker.put_tracking_in_frame_with_regions_with_couting(frame)
        cv2.imshow("tracking with data showing", frame_with_tracking_regions_and_counting)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    # save data counted
    save_file = args.save_filepath
    tracker.save_data(save_file)





