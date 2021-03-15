import sys
import os
import json
import logging
import argparse
from utils.utils import *
from utils.log import logger
from utils.timer import Timer
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils import visualization as vis
from track import eval_seq

from tracker.multitracker import JDETracker

sys.path.append("/usr/local/python")

from openpose import pyopenpose as op


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="cfg/yolov3_1088x608.cfg", help="cfg file path")
    parser.add_argument("--weights", type=str, default=None, help='path to weights file')
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--min_box_area", type=float, default=200, help="filter out tiny boxes")
    parser.add_argument("--track_buffer", type=int, default=30, help="tracking buffer")
    parser.add_argument("--input_video", type=str, default=None, help="path to the input video")
    parser.add_argument("--output_dir", type=str, default=None, help="expected output root path")

    parser.add_argument("--openpose_dir", type=str, default=None)
    parser.add_argument("--model_pose", type=str, default="BODY_25", help='path to the weights file')
    parser.add_argument("--output_json_dir", type=str, default="MMAct_annotator")
    parser.add_argument("--hand_pose", type=str2bool, default=False)
    parser.add_argument("--head_pose", type=str2bool, default=False)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_gpu_start", type=int, default=0)
    parser.add_argument("--num_people_max", type=int, default=2)
    parser.add_argument("--scale_number", type=int, default=4)
    parser.add_argument("--scale_gap", type=float, default=0.25)
    parser.add_argument("--net_resolution", type=str, default="-1x368")
    parser.add_argument("--render_threshold", type=float, default=0.5)

    args = parser.parse_known_args()
    print("Args: {}".format(args))

    return args


if __name__ == '__main__':
    args = get_args()

    command_args = args[0]

    logger.setLevel(logging.INFO)

    if not command_args.output_dir == None:
        if not os.path.exists(command_args.output_dir):
            os.makedirs(command_args.output_dir, exist_ok=True)

    if not os.path.exists(command_args.output_json_dir):
        os.makedirs(command_args.output_json_dir, exist_ok=True)

    out_json_filename = os.path.join(command_args.output_json_dir, os.path.basename(command_args.input_video).split(".")[0] + ".json")

    params = dict()
    params["model_folder"] = os.path.join(command_args.openpose_dir, "models")
    params["model_pose"] = command_args.model_pose
    params["face"] = command_args.head_pose
    params["hand"] = command_args.hand_pose
    params["net_resolution"] = command_args.net_resolution
    params["scale_number"] = command_args.scale_number
    params["scale_gap"] = command_args.scale_gap
    params["num_gpu"] = command_args.num_gpu
    params["num_gpu_start"] = command_args.num_gpu_start
    params["number_people_max"] = command_args.num_people_max
    params["display"] = command_args.display
    params["render_threshold"] = command_args.render_threshold

    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cfg_dict = parse_model_cfg(path=command_args.cfg)
    command_args.img_size = [int(cfg_dict[0]["width"]), int(cfg_dict[0]["height"])]
    timer = Timer()

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(command_args.input_video, command_args.img_size)
    frame_rate = dataloader.frame_rate
    orig_width = dataloader.vw
    orig_height = dataloader.vh
    width = dataloader.w
    height = dataloader.h

    video_writer = cv2.VideoWriter(os.path.join(command_args.output_dir, os.path.splitext(os.path.basename(command_args.input_video))[0] + ".mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), float(frame_rate), (orig_width, orig_height))

    tracker = JDETracker(opt=command_args, frame_rate=frame_rate)

    video_info = {}
    frame_id = 0
    for path, img, img0, orig_img in dataloader:
        frame_info = {}
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            datum = op.Datum()
            tlwh = t.tlwh
            tid = t.track_id
            x, y, w, h = tlwh
            x = (orig_width / command_args.img_size[0]) * x
            y = (orig_height / command_args.img_size[1]) * y
            w = (orig_width / command_args.img_size[0]) * w
            h = (orig_height / command_args.img_size[1]) * h
            image_to_detected = orig_img[max(0, int(y - 0.2 * h)): min(int(y + h + 0.2 * h), orig_height - 1),
                                         max(0, int(x - 0.2 * w)): min(int(x + w + 0.2 * w), orig_width - 1)].copy()
            datum.cvInputData = image_to_detected
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            poseKeypoints = datum.poseKeypoints

            if poseKeypoints is None:
                print("No detected poses")
                continue
            else:
                frame_info[str(tid)] = poseKeypoints[0]
                tlwh = (x, y, w, h)
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > command_args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            detected_cropped_image = datum.cvOutputData.copy()
            orig_img[max(0, int(y - 0.2 * h)): min(int(y + h + 0.2 * h), orig_height - 1),
                     max(0, int(x - 0.2 * w)): min(int(x + w + 0.2 * w), orig_width - 1)] = detected_cropped_image.copy()

        timer.toc()
        online_im = vis.plot_tracking(image=orig_img, tlwhs=online_tlwhs, obj_ids=online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)

        video_info[str(frame_id)] = frame_info
        video_writer.write(online_im)
        frame_id += 1

    with open(out_json_filename, "w+") as f:
        json.dump(video_info, f, indent=3, cls=NumpyEncoder)
    video_writer.release()
    print("Saved video")
    print("Finished")
