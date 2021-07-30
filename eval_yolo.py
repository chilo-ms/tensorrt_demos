"""eval_yolo.py

This script is for evaluating mAP (accuracy) of YOLO models.
"""


import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import yolo_cls_to_ssd



HOME = os.environ['HOME']
VAL_IMGS_DIR = HOME + '/data/coco/images/val2017'
VAL_ANNOTATIONS = HOME + '/data/coco/annotations/instances_val2017.json'


def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs_dir', type=str, default=VAL_IMGS_DIR,
        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument(
        '--annotations', type=str, default=VAL_ANNOTATIONS,
        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument(
        '--non_coco', action='store_true',
        help='don\'t do coco class translation [False]')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)

def parse_annotations(filename):
    import json
    annotations = {}
    with open(filename, 'r') as f:
        annotations = json.load(f)

    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id

def generate_results(trt_yolo, imgs_dir, jpgs, results_file, non_coco, annotations):
    """Run detection on each jpg and write results to file."""
    results = []
    img_name_to_img_id = parse_annotations(annotations)
    # for jpg in progressbar(jpgs):
    for jpg in jpgs:
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        # image_id = int(jpg.split('.')[0].split('_')[-1])
        image_id = img_name_to_img_id[jpg]

        # boxes, confs, clss = trt_yolo.detect(img, conf_th=0.35)
        # for box, conf, cls in zip(boxes, confs, clss):
            # x = float(box[0])
            # y = float(box[1])
            # w = float(box[2] - box[0] + 1)
            # h = float(box[3] - box[1] + 1)
            # cls = int(cls)
            # cls = cls if non_coco else yolo_cls_to_ssd[cls]
            # if cls == 1:
                # results.append({'image_id': image_id,
                                # # 'category_id': cls,
                                # 'category_id': 0,
                                # 'bbox': [x, y, w, h],
                                # 'score': float(conf)})

        boxes, confs = trt_yolo.detect(img, conf_th=0.35)
        for box, conf in zip(boxes, confs):
                results.append({'image_id': image_id,
                                'category_id': 0, # only person
                                'bbox': box,
                                'score': float(conf)})
    # print(results)
    # with open(results_file, 'w') as f:
        # f.write(json.dumps(results, indent=4))

    return results


def main():
    args = parse_args()
    check_args(args)
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    results_file = 'yolo/results_%s.json' % args.model

    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    results = generate_results(trt_yolo, args.imgs_dir, jpgs, results_file,
                     non_coco=args.non_coco, annotations=args.annotations)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    # cocoDt = cocoGt.loadRes(results_file)
    cocoDt = cocoGt.loadRes(results)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
