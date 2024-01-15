import os
import argparse
import collections
import itertools
import logging
import multiprocessing
from pathlib import Path
import random
import json

import numpy as np
from natsort import natsorted
from PIL import Image
from pycocotools.coco import COCO
from script_utils.common import common_setup
import cv2
from tqdm import tqdm
from collections import defaultdict

import tao.utils.vis as vis_utils
from tao.utils import fs
from tao.utils.coco import interpolate_annotations
from tao.utils.colormap import colormap
from tao.utils.misc import parse_bool
from tao.utils.video import video_writer
from tao.utils.load_prediction import Prediction
from utils import get_video_name, default_arg_parser, clip_annotation

def visualize_star(x):
    return visualize(*x)


def visualize(coco, video: str, labeled_frames, args, tao_amodal_prediction, tao_amodal_prediction2):
    separator = args.separator_width

    output_video = args.output_dir / (video + '.mp4')
    frames_dir = args.images_dir / video
    if not frames_dir.exists():
        logging.warn(f"Could not find images at {frames_dir}")
        return

    frame_infos = {
        x['file_name'].split('/')[-1]: x
        for x in labeled_frames
    }
    print(frames_dir)
    frames = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))

    # resolve all symlinks and correct the argo links
    for i, frame in enumerate(frames):
        if os.path.islink(frame):
            real_path = os.readlink(frame)
            real_path = Path(str(real_path).replace('ArgoVerse1.1', 'Argoverse-1.1'))
            frames[i] = real_path
        

    width, height = Image.open(frames[0]).size
    expected_w = labeled_frames[0]['width']
    expected_h = labeled_frames[0]['height']
    assert width == expected_w, (
        f'Visualization image width ({width}) does not match width '
        f'specified in annotation ({expected_w})')
    assert height == expected_h, (
        f'Visualization image height ({height}) does not match width '
        f'specified in annotation ({expected_h})')
    output_video.parent.mkdir(exist_ok=True, parents=True)

    if args.use_tracks:
        color_generator = itertools.cycle(colormap(rgb=True).tolist())
        color_map = collections.defaultdict(lambda: next(color_generator))

    frame_annotations = {
        frame.rsplit('.', 1)[0]: coco.imgToAnns[info['id']]
        for frame, info in frame_infos.items()
    }

    # Get predictions.
    frame_predictions = {
        frame.rsplit('.', 1)[0]: tao_amodal_prediction.imgToAnns[info['id']]
        for frame, info in frame_infos.items()
    }
    frame_predictions2 = {
        frame.rsplit('.', 1)[0]: tao_amodal_prediction2.imgToAnns[info['id']]
        for frame, info in frame_infos.items()
    }

    if args.skip_unknown_categories:
        for frame, anns in frame_annotations.items():
            frame_annotations[frame] = [
                x for x in anns
                if coco.cats[x['category_id']]['synset'] != 'unknown'
            ]

            frame_predictions[frame] = [
                x for x in anns
                if coco.cats[x['category_id']]['synset'] != 'unknown'
            ]

            frame_predictions2[frame] = [
                x for x in anns
                if coco.cats[x['category_id']]['synset'] != 'unknown'
            ]

    # Trim to only show labeled segment.
    first = next(i for i, f in enumerate(frames)
                 if f.stem in frame_annotations)
    last = next(i for i, f in reversed(list(enumerate(frames)))
                if f.stem in frame_annotations)

    frames = frames[first:last+1]
    interpolated_annotations = {}
    if args.use_tracks and args.interpolate and args.speed_up > 0:
        interpolated_annotations = interpolate_annotations(
            [x.stem for x in frames], frame_annotations)

    unlabeled_location = None
    if (args.original_location == 'top'
            or (args.original_location == 'auto' and width > height)):
        unlabeled_location = 'top'
        height = height * 2 + separator
        black_line = np.zeros((separator, width, 3), dtype=np.uint8)
    elif (args.original_location == 'left'
            or (args.original_location == 'auto' and height > width)):
        unlabeled_location = 'left'
        width = width * 2 + separator
        black_line = np.zeros((height, separator, 3), dtype=np.uint8)

    cats = coco.cats.copy()
    for cat in cats.values():
        if cat['name'] == 'baby':
            cat['name'] = 'person'

    with video_writer(str(output_video), (width*2, height)) as writer:
        # Randomly clip video lengths to a certain length.
        if args.clip_video_length:
            first_frame_start, first_frame_end = 0, max(len(frames) - args.clip_video_length, 0)
            first_frame = random.randint(first_frame_start, first_frame_end)
            frames = frames[first_frame: first_frame + args.clip_video_length]
        
        for i, frame in enumerate(frames):
            is_annotated = frame.stem in frame_annotations
            is_interpolated = frame.stem in interpolated_annotations
            
            # Ignore those unlabeled frames when speed_up < 0.
            if ((not is_annotated)
                    and (args.speed_up < 0 or (i % args.speed_up) != 0)):
                continue


            frame = Path(str(frame).replace('ArgoVerse1.1', 'Argoverse-1.1'))
            image = np.array(Image.open(frame))
            full_image = np.ones((int(height*1.5), int(width*3.0), 3), dtype=np.uint8) * 255
            startx = int(width/4)
            endx = startx + width
            starty = int(height/5)
            endy = starty + height
            full_image[starty:endy, startx:endx, :] = image

            '''
            * Show Image ID.
            '''
            if is_annotated:
                annotations = frame_annotations[frame.stem]
                predictions = frame_predictions[frame.stem]
                if annotations:
                    image_id = annotations[0]['image_id']
                elif predictions:
                    image_id = predictions[0]['image_id']
                else:
                    image_id = "/".join(str(frame).split("/")[-3:])
                if args.show_image_id:
                    full_image = vis_utils.vis_class(full_image, [int(startx + (endx - startx)* 0.4 ), starty // 2], str(image_id),
                                    bg_color=(255, 255, 255),
                                    text_color=(0, 0, 0),
                                    font_scale=2.5,
                                    thickness=3)
        
            # Draw the raw image for the second tracker predictions.
            full_image[starty:endy, 
                    int(width*1.5+startx): int(width*1.5+endx)] = image
    
                
            '''
            * Plot the amodal annotation
            '''
            if is_annotated or is_interpolated:
                if is_interpolated:
                    annotations = interpolated_annotations[frame.stem]
                else:
                    annotations = frame_annotations[frame.stem]
                    predictions = frame_predictions[frame.stem]
                    predictions2 = frame_predictions2[frame.stem]

                if args.use_tracks:
                    prediction_colors = [
                        color_map[x['track_id']] for x in predictions
                    ]
                    prediction_colors2 = [
                        color_map[x['track_id']] for x in predictions2
                    ]
                    if args.color:
                        prediction_colors = [args.color for _ in prediction_colors]
                        prediction_colors2 = [args.color for _ in prediction_colors2]

                if args.filter_tracks:
                    prediction_colors = [color for i, color in enumerate(prediction_colors) if predictions[i]['track_id'] in args.filter_tracks]
                    prediction_colors2 = [color for i, color in enumerate(prediction_colors2) if predictions2[i]['track_id'] in args.filter_tracks]
                    predictions = [pred for pred in predictions if pred['track_id'] in args.filter_tracks]
                    predictions2 = [pred for pred in predictions2 if pred['track_id'] in args.filter_tracks]
                
                # Hyperparameters for bounding box and font.
                opacity = -1
                thickness = 5
                font_scale = 1.0
                font_thickness = 2
                
                if args.clip_annotation:
                    for x in annotations:
                        clip_annotation(x, image)

                visualized = full_image.copy()
                if args.transparent:
                    visualized[:, :(visualized.shape[1] // 2)] = vis_utils.transparent_except_bbox(visualized[:, :(visualized.shape[1]  // 2)], predictions, oy=starty,
                                                          ox=startx, modal=True, opacity=0.4)
                    visualized[:, (visualized.shape[1] // 2):] = vis_utils.transparent_except_bbox(visualized[:, (visualized.shape[1] // 2):], predictions2, oy=starty,
                                                          ox=startx, modal=True, opacity=0.4)
                     
                # Draw the predicted bounding boxes
                visualized = vis_utils.overlay_amodal_boxes_prediction(visualized,
                                                          predictions,
                                                          oy=starty,
                                                          ox=startx,
                                                          colors=prediction_colors,
                                                          thickness=thickness,
                                                          fill_opacity=opacity)

                visualized = vis_utils.overlay_amodal_boxes_prediction(visualized,
                                                          predictions2,
                                                          oy=starty,
                                                          ox=int(width*1.5+startx),
                                                          colors=prediction_colors2,
                                                          thickness=thickness,
                                                          fill_opacity=opacity)

                if args.show_categories:
                    visualized = vis_utils.overlay_amodal_class_prediction(
                        visualized,
                        predictions,
                        oy=starty,
                        ox=startx,
                        categories=cats,
                        background_colors=prediction_colors,
                        font_scale=font_scale,
                        font_thickness=font_thickness,
                        show_track_id=args.show_track_id)
                    
                    visualized = vis_utils.overlay_amodal_class_prediction(
                        visualized,
                        predictions2,
                        oy=starty,
                        ox=int(width*1.5+startx),
                        categories=cats,
                        background_colors=prediction_colors2,
                        font_scale=font_scale,
                        font_thickness=font_thickness,
                        show_track_id=args.show_track_id) 

                if unlabeled_location == 'top':
                    visualized = np.concatenate(
                        [full_image, visualized])

                elif unlabeled_location == 'left':
                    visualized = np.concatenate(
                        [full_image, visualized], axis=1)

                for _ in range(
                        args.slow_down if not is_interpolated else 1):
                    visualized = cv2.resize(visualized, (width*2, height))
                    writer.write_frame(visualized)
            else:
                if unlabeled_location == 'top':
                    full_image = np.concatenate([full_image, full_image])
                elif unlabeled_location == 'left':
                    full_image = np.concatenate([full_image, full_image], axis=1)
                full_image = cv2.resize(full_image, (width*2, height))
                writer.write_frame(full_image)

def main():
    '''
    * Process Arguments
    '''
    args = default_arg_parser()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    '''
    * Create Dataset to load videos
    '''
    coco = COCO(args.annotations)
    tao_amodal_prediction = Prediction(args.predictions, args.score_threshold)
    tao_amodal_prediction2 = Prediction(args.predictions2, args.score_threshold)
    
    '''
    * Get the video names we want.
    '''
    get_video_name(args)

    '''
    * Select the videos we need
    '''
    all_videos = set()
    videos = collections.defaultdict(list)
    for image in coco.imgs.values():
        if 'video' in image:
            video = str(Path(image['video']).with_suffix(''))
        else:
            video = image['file_name'].split('/')[-2]
        all_videos.add(video)

        if args.video_name is None or video in args.video_name or args.video_name[0] in video:
            videos[video].append(image)
    tasks = []
    for video, labeled_frames in videos.items():
        output_video = args.output_dir / (video + '.mp4')
        tasks.append((coco, video, labeled_frames, args, tao_amodal_prediction, tao_amodal_prediction2))

    '''
    * Visualization with Multi-Processing
    '''
    if args.workers == 0:
        for task in tqdm(tasks):
            visualize(*task)
    else:
        pool = multiprocessing.Pool(args.workers)
        import random
        random.shuffle(tasks)
        list(tqdm(pool.imap_unordered(visualize_star, tasks),
                  total=len(tasks)))


if __name__ == "__main__":
    main()
