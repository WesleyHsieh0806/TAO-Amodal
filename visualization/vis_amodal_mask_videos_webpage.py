import os
import argparse
import collections
import itertools
import logging
import multiprocessing
from pathlib import Path
import random

import numpy as np
from natsort import natsorted
from PIL import Image
from pycocotools.coco import COCO
from script_utils.common import common_setup
import cv2
from tqdm import tqdm

import tao.utils.vis as vis_utils
from tao.utils import fs
from tao.utils.coco import interpolate_annotations
from tao.utils.colormap import colormap
from tao.utils.misc import parse_bool
from tao.utils.video import video_writer
from burstapi import BURSTDataset
import burstapi.visualization_utils as viz_utils
from utils import get_video_name, default_arg_parser, clip_annotation

def visualize_star(x):
    return visualize(*x)


def visualize(coco, video: str, labeled_frames, args, burst_video):
    # Ensure amodal/mask annotations correspond to the same video
    assert (video in burst_video.name) or (burst_video.name in video), ("videos of Amodal Annotations and mask annotations do not align!")
    separator = args.separator_width

    output_video = args.output_dir / (video + '.mp4')
    # if output_video.exists():
    #     logging.info(f'{output_video} exists, skipping')
    #     return
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
    if args.skip_unknown_categories:
        for frame, anns in frame_annotations.items():
            frame_annotations[frame] = [
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
    if args.use_tracks and args.interpolate:
        interpolated_annotations = interpolate_annotations(
            [x.stem for x in frames], frame_annotations, modal=args.modal)

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

    nof_annotated = 0
    min_track = float('inf')
    with video_writer(str(output_video), (width, height)) as writer:
        if args.clip_video_length:
            for frame in frames:
                if frame.stem not in frame_annotations:
                    continue
                annotations = frame_annotations[frame.stem]
                if annotations:
                    min_track = min([x['track_id'] for x in annotations])
                    break
            first_frame_start, first_frame_end = 0, max(len(frames) - args.clip_video_length, 0)
            first_frame = random.randint(first_frame_start, first_frame_end)
            frames = frames[first_frame: first_frame + args.clip_video_length]
        
        for i, frame in enumerate(frames):

            is_mask_annotated = burst_video.is_mask_annotated(frame)
            is_annotated = frame.stem in frame_annotations
            is_interpolated = frame.stem in interpolated_annotations
            
            if ((not is_mask_annotated) and (not is_annotated)
                    and (args.speed_up < 0 or (i % args.speed_up) != 0)):
                # ignore those unlabeled frames
                continue
            # if ((not is_mask_annotated) and (not is_annotated)):
                # continue
            frame = Path(str(frame).replace('ArgoVerse1.1', 'Argoverse-1.1'))
            image = np.array(Image.open(frame))
            full_image = np.ones((height*2, width*2, 3), dtype=np.uint8) * 255
            startx = int(width/2)
            endx = startx + width
            starty = int(height/2)
            endy = starty + height
            full_image[starty:endy, startx:endx, :] = image
            
            # plot GT.
            if is_annotated:
                annotations = frame_annotations[frame.stem]
                if min_track == float('inf') and annotations:
                    min_track = min([x['track_id'] for x in annotations])

                if annotations:
                    image_id = annotations[0]['image_id']
                else:
                    image_id = "/".join(str(frame).split("/")[-3:])
                if args.show_image_id:
                    full_image = vis_utils.vis_class(full_image, [int(startx), starty // 2], str(image_id),
                                    bg_color=(255, 255, 255),
                                    text_color=(0, 0, 0),
                                    font_scale=2.5,
                                    thickness=3)
                
            '''
            * Plot the mask annotation
            '''
            if is_mask_annotated:
                mask_t = burst_video.get_mask_by_frame(frame)

                for track_id in mask_t:
                    color = color_map[track_id]
                    if args.color:
                        color = args.color
                    if args.filter_tracks and (track_id + min_track - 1) not in args.filter_tracks:
                        continue
                    full_mask = np.zeros((1, height*2, width*2), dtype=np.uint8)
                    full_mask[:, starty:endy, startx:endx] = mask_t[track_id].astype(np.uint8)
                    full_image = vis_utils.vis_mask(full_image,
                        full_mask,
                        color,
                        alpha=0.4,
                        show_border=True,
                        border_alpha=0.5,
                        border_thick=1,
                        border_color=None)

                
            '''
            * Plot the amodal annotation
            '''
            if is_annotated or is_interpolated:
                if is_interpolated:
                    annotations = interpolated_annotations[frame.stem]
                else:
                    annotations = frame_annotations[frame.stem]
                colors = None
                if args.use_tracks:
                    colors = [
                        color_map[x['track_id']  - min_track + 1] for x in annotations
                    ]
                    if args.color:
                        colors = [args.color for color in colors]
                if args.filter_tracks:
                    colors = [color for i, color in enumerate(colors) if annotations[i]['track_id'] in args.filter_tracks]
                    annotations = [ann for ann in annotations if ann['track_id'] in args.filter_tracks]
                opacity = 0.65
                thickness = 3
                font_scale = 1.0
                font_thickness = 2
                
                if args.clip_annotation:
                    for x in annotations:
                        clip_annotation(x, image)
                
                visualized = full_image.copy()
                if args.transparent:
                    visualized = vis_utils.transparent_except_bbox(visualized, annotations, modal=args.modal, opacity=0.25)
                
                # draw the bounding box
                if not args.modal:
                    visualized = vis_utils.overlay_amodal_boxes_coco(visualized,
                                                          annotations,
                                                          colors=colors,
                                                          thickness=thickness,
                                                          fill_opacity=opacity)
                else:
                    colors = [color for i, color in enumerate(colors) if 'bbox' in annotations[i]]
                    visualized = vis_utils.overlay_modal_boxes_coco(visualized,
                                                          annotations,
                                                          colors=colors,
                                                          thickness=thickness,
                                                          fill_opacity=opacity)
                if args.show_categories:
                    visualized = vis_utils.overlay_amodal_class_coco(
                        visualized,
                        annotations,
                        categories=cats,
                        font_scale=font_scale,
                        background_colors=colors,
                        font_thickness=font_thickness,
                        show_track_id=args.show_track_id)
                elif args.show_visibility:
                    # shows the visibility of each amodal bounding box
                    visualized = vis_utils.overlay_amodal_visibility_coco(
                        visualized,
                        annotations,
                        categories=cats,
                        font_scale=font_scale,
                        font_thickness=font_thickness,
                        show_track_id=args.show_track_id)

                if unlabeled_location == 'top':
                    visualized = np.concatenate(
                        [image, black_line, visualized])
                elif unlabeled_location == 'left':
                    visualized = np.concatenate(
                        [image, black_line, visualized], axis=1)
                for _ in range(
                        args.slow_down if not is_interpolated else 1):
                    visualized = cv2.resize(visualized, (width, height))
                    writer.write_frame(visualized)
                # for _ in range(
                #         args.slow_down):
                #     visualized = cv2.resize(visualized, (width, height))
                #     writer.write_frame(visualized)
            else:
                if unlabeled_location == 'top':
                    full_image = np.concatenate([full_image, full_image])
                elif unlabeled_location == 'left':
                    full_image = np.concatenate([full_image, full_image], axis=1)
                # for _ in range(2):
                #     full_image = cv2.resize(full_image, (width, height))
                #     writer.write_frame(full_image)
                full_image = cv2.resize(full_image, (width, height))
                writer.write_frame(full_image)


def main():
    # Use first line of file docstring as description if it exists.
    args = default_arg_parser()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    '''
    * Create Dataset to load videos
    '''
    coco = COCO(args.annotations)
    dataset = BURSTDataset(annotations_file=args.mask_annotations,
                           images_base_dir=args.images_dir)
    '''
    * Get the video names that we want to visualize
    '''
    get_video_name(args)

    '''
    * Select the videos we need
    '''
    all_videos = set()
    videos = collections.defaultdict(list)
    for image in coco.imgs.values():
        # print(image)
        # if 'LaSOT' in image:
        #     continue
        if 'video' in image:
            video = str(Path(image['video']).with_suffix(''))
        else:
            video = image['file_name'].split('/')[-2]
        all_videos.add(video)

        if args.video_name is None or video in args.video_name or args.video_name[0] in video:
            # video example: train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e
            videos[video].append(image)
    tasks = []
    for video, labeled_frames in videos.items():
        # print(video, labeled_frames)
        # if 'LaSOT' in video:
        #     continue
        output_video = args.output_dir / (video + '.mp4')
        # if output_video.exists():
        #     logging.info(f'{output_video} exists, skipping')
        #     continue
        tasks.append((coco, video, labeled_frames, args, dataset.get_video_by_name(video)))

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
