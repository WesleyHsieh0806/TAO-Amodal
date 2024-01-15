import os 
import numpy as np
import json 
import argparse 
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from tao.utils.misc import parse_bool
def default_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input/Output parameters
    parser.add_argument('--annotations', type=Path, required=True,
                        help='Path to TAO-Amodal annotation json.')
    parser.add_argument('--mask-annotations', type=Path, required=False,
                        help='The path to your BURST annotation json.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output folder where you want to save your visualization results.')
    parser.add_argument('--images-dir', type=Path, required=True,
                        help=('Path to TAO-Amodal/frames. '
                         'Make sure you download all the frames following instructions'
                         ' at https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal'
                         ' before using the visualization script.'))
    # Used in random quality check
    parser.add_argument('--random-quality-check', type=parse_bool, default=False)
    parser.add_argument('--random-quality-check-size', type=int, default=7)
    parser.add_argument('--split', type=str, 
                        help="train/val/test. Videos will be randomly picked from each subset in this split and visualized.")
    
    # Used in selected quality check
    parser.add_argument('--selected-quality-check', type=parse_bool, default=False,
                        help='If True, we select videos with more heavy occlusions for visualization.')
    parser.add_argument('--selected-quality-check-size', type=int, default=7)
    parser.add_argument('--video-name', type=str, nargs='*',
                        help='If specified, only specified videos will be visualized.')

    # Video settings.
    parser.add_argument('--clip-video-length', default=None, type=int, 
                        help="If this number is given, we randomly visualize a small interval of the video to control the video length.")
    parser.add_argument('--show-image-id', default=True, type=parse_bool,
                        help='If True, the ```image_id``` of each frame will be displayed at \
                            the top of the video')
    parser.add_argument(
        '--original-location',
        default='none',
        choices=['none', 'top', 'left', 'auto'],
        help=('Draw the original frame to the left or to the top of the '
              'labeled frame.'))
    parser.add_argument('--speed-up',
                        type=int,
                        default=2,
                        help=('How much to speed up unlabeled frames. If '
                              'set to -1, skip unlabeled frames.'))
    parser.add_argument('--slow-down',
                        type=int,
                        default=15,
                        help='How much to slow down labeled frames.')
    parser.add_argument('--separator-width', default=5, type=int)
    
    # Bounding box settings.
    parser.add_argument('--show-categories', default=True, type=parse_bool,
                        help='If True, we show the category name of each track.')
    parser.add_argument('--show-visibility', default=False, type=parse_bool, 
                        help="Show the Visibility of Amodal Bounding Boxes, \
                            This property only works when show-categories is false")
    parser.add_argument('--show-track-id', type=parse_bool, default=False,
                        help='If True, the track id will be displayed after the category/visibility. \
                            This attribute works only when either show-categories or show-visibility is True')
    parser.add_argument('--interpolate', default=False, type=parse_bool,
                        help='If True, we visualize amodal boxes in non-annotated frames using interpolation.')
    parser.add_argument('--transparent', default=False, type=parse_bool, 
                        help="If true, we make the background transparent to emphasize the bounding boxes")
    parser.add_argument('--modal', default=False, type=parse_bool, 
                        help="If true, we visualize the modal box instead of the amodal boxes")
    parser.add_argument('--color', default=None, type=int, nargs="+", 
                        help=('If specified, we use this color as the bounding box border. '
                            'This parameter is commonly used when we want to visualize a certain track.'))
    parser.add_argument('--filter-tracks', type=int, nargs='*', 
                        help="If specified, we only visualize the specified track ids.")
    parser.add_argument('--clip-annotation', type=parse_bool, default=False, 
                        help="If True, we clip the out-of-frame boxes")
    parser.add_argument('--skip-unknown-categories',
                        type=parse_bool,
                        default=False)
    
    parser.add_argument('--use-tracks', type=parse_bool, default=True)

    
    # Used in visualizing Predictions
    parser.add_argument('--predictions', type=Path, required=False, help="Path of tracker prediction results, e.g., track.json")
    parser.add_argument('--predictions2', type=Path, required=False, help="Path of second tracker prediction results, e.g., track.json")
    parser.add_argument('--score-threshold', type=float, default=0.5)


    
    # Amodal Expander Visualization relevant hyperparameters
    parser.add_argument('--expand-ratio', type=float, default=1.1)  # Only modal boxes whose area are enlarged by 1.1 times at least will be visualized
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers to run multi-processing.')

    args = parser.parse_args()
    return args

def get_video_name(args):
    if (int(args.video_name is not None) + int(args.random_quality_check) + int(args.selected_quality_check)) > 1:
        raise ValueError("You can only choose one mode 1. specifying video name 2. random quality check 3. selected quality check")
    if (not args.random_quality_check) and (not args.selected_quality_check) and (len(args.video_name) == 0):
        raise ValueError("Please specify the video name you want to visualize")
    
    '''
    * If we need to conduct random-quality check, we randomly select 7 videos from each split
    '''
    np.random.seed(0)
    if args.random_quality_check:
        # TODO
        args.video_name = []
        image_split_dir = os.path.join(args.images_dir, args.split)
        
        # Traverse different dataset, for each dataset, we select 7 videos
        print("Randomly select {} videos from each dataset...".format(args.random_quality_check_size))
        add_random_check_videos(image_split_dir, args)

    '''
    * If we want to do selected quality check
    '''
    if args.selected_quality_check:
        args.video_name = select_video(args.annotations, args)

def add_random_check_videos(image_split_dir, args):
    for dataset_path in os.listdir(image_split_dir):
        selected_videos = np.random.choice(list(os.listdir(os.path.join(image_split_dir, dataset_path))), size = args.random_quality_check_size, replace=False)
        for video in selected_videos:
            args.video_name.append(os.path.join(args.split, dataset_path, video))


def select_video(annotations, args, threshold=0.5):
    '''
    * Annotations: str of the path to TAO-Amodal annotation
        Select top-7 videos in each sub-dataset which contains most tracks with visibility <= threshold
    * Output:
        a list that contains all the video paths
    '''
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
        
    with open(annotations, 'r') as f:
        ann_dict = json.load(f)
    
    # first, we map video id to video name and dataset
    vid_to_vname = defaultdict(str)
    dataset_to_vids = defaultdict(list)
    print("Map Video Id to name...")
    for i, video in enumerate(ann_dict["videos"]):
        print("[{}/{}]".format(i + 1, len(ann_dict["videos"])), end='\r')
        vid = video["id"]
        vname = video["name"]
        dataset = video["metadata"]["dataset"]

        vid_to_vname[vid] = vname
        dataset_to_vids[dataset].append(vid)
    print()

    selected_videos = defaultdict(list)
    vid_to_nof_occluded_tracks = defaultdict(int)
    
    # Compute the visibility and record the nof tracks whose visibility <= threshold
    print("Compute Visibility for tracks...")
    for i, ann in enumerate(ann_dict["annotations"]):
        print("[{}/{}]".format(i + 1, len(ann_dict["annotations"])), end='\r')
        amodal_box = ann["amodal_bbox"]
        if "bbox" in ann:
            bbox = ann["bbox"]
            visibility = bb_intersection_over_union([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], 
                        [amodal_box[0], amodal_box[1], amodal_box[0] + amodal_box[2], amodal_box[1] + amodal_box[3]])
        elif "segmentation" in ann:
            segmentation = ann["segmentation"][0]
            visibility = bb_intersection_over_union([segmentation[0], segmentation[1], segmentation[4], segmentation[5]], 
                        [amodal_box[0], amodal_box[1], amodal_box[0] + amodal_box[2], amodal_box[1] + amodal_box[3]])
        else:
            visibility = 0.0

        if visibility <= threshold:
            vid_to_nof_occluded_tracks[ann["video_id"]] += 1
    print()


    # select top-7 videos from each datasets that contains most number of tracks with visibility <= threshold
    print("Select Videos for each dataset...")
    for dataset in tqdm(dataset_to_vids):
        selected_videos[dataset] = sorted(dataset_to_vids[dataset], key = lambda vid: -vid_to_nof_occluded_tracks[vid])[:args.selected_quality_check_size]
        selected_videos[dataset] = [vid_to_vname[vid] for vid in selected_videos[dataset]]

    return [vname for dataset in selected_videos for vname in selected_videos[dataset]]


def clip_annotation(x, image):
    '''
    * x: a dict contains the "amodal_bbox" attribute
    * Modify x implace
    '''
    x["amodal_bbox"] = [max(x['amodal_bbox'][0], 0), max(x['amodal_bbox'][1], 0), 
                                            min(x['amodal_bbox'][0] + x['amodal_bbox'][2], image.shape[1]), 
                                            min(x['amodal_bbox'][1] + x['amodal_bbox'][3], image.shape[0])]
    x["amodal_bbox"][2] -= x['amodal_bbox'][0]  # back to XYWH format
    x["amodal_bbox"][3] -= x["amodal_bbox"][1]  # back to XYWH format