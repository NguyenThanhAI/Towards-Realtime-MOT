import os
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

    args = parser.parse_args()
    print("Args: {}".format(args))

    return args


if __name__ == '__main__':
    args = get_args()

    logger.setLevel(logging.INFO)

    if not args.output_dir == None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    cfg_dict = parse_model_cfg(path=args.cfg)
    args.img_size = [int(cfg_dict[0]["width"]), int(cfg_dict[0]["height"])]
    timer = Timer()

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(args.input_video, args.img_size)
    frame_rate = dataloader.frame_rate
    orig_width = dataloader.vw
    orig_height = dataloader.vh
    width = dataloader.w
    height = dataloader.h

    video_writer = cv2.VideoWriter(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_video))[0] + ".mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), float(frame_rate), (orig_width, orig_height))

    tracker = JDETracker(opt=args, frame_rate=frame_rate)

    frame_id = 0
    for path, img, img0, orig_img in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            x, y, w, h = tlwh
            x = (orig_width / args.img_size[0]) * x
            y = (orig_height / args.img_size[1]) * y
            w = (orig_width / args.img_size[0]) * w
            h = (orig_height / args.img_size[1]) * h
            tlwh = (x, y, w, h)
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        online_im = vis.plot_tracking(image=orig_img, tlwhs=online_tlwhs, obj_ids=online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)

        video_writer.write(online_im)
        frame_id += 1

    video_writer.release()
