from argparse import ArgumentParser
from tqdm import tqdm

from burstapi.dataset import BURSTDataset

import cv2
import numpy as np
import burstapi.visualization_utils as viz_utils


def main(args):
    dataset = BURSTDataset(annotations_file=args.annotations_file,
                           images_base_dir=args.images_base_dir)

    cv2.namedWindow("Video Frame", cv2.WINDOW_NORMAL)
    color_map = viz_utils.create_color_map()

    print("-----------------------------------------\n"
          "        NAVIGATION INSTRUCTIONS\n"
          "-----------------------------------------\n"
          "Next frame:     'D' OR right-arrow key\n"
          "Previous frame: 'A' OR left-arrow key\n"
          "(N)ext video:   'N'\n"
          "(Q)uit:         'Q'\n"
          "-----------------------------------------\n")

    for i in range(dataset.num_videos):
        video = dataset[i]

        print(f"- Dataset: {video.dataset}\n"
              f"- Name: {video.name}")

        video_stats = video.stats()

        for k, v in video_stats.items():
            print(f"- {k}: {v}")

        # keys: track IDs, values: category IDs
        category_ids = video.track_category_ids

        # load all images as a list (length T) of numpy arrays with shape [H, W, 3]
        images = video.load_images()

        if args.first_frame_annotations:
            annotations = video.load_first_frame_annotations()
        else:
            # load all masks as a list (length T) of lists (length N) of numpy arrays of shape [H, W]. (N = number of object
            # tracks in the video)
            annotations = video.load_masks()

        print("\nGenerating visualizations...")
        annotated_images = []

        for image_t, annotations_t in tqdm(zip(images, annotations), total=len(images)):
            image_t = image_t[:, :, ::-1]  # convert from RGB to BGR for OpenCV

            for track_id, annotation in annotations_t.items():

                if isinstance(annotation, np.ndarray):  # mask object
                    mask = annotation
                    point = None
                else:
                    mask = annotation["mask"]
                    point = (annotation["point"]["x"], annotation["point"]["y"])

                if not np.any(mask):  # all zeros mask
                    continue

                text_label = dataset.category_names[category_ids[track_id]]
                image_t = viz_utils.annotate_image(
                    image_t, mask, color=color_map[track_id % 256], label=text_label, point=point
                )

            annotated_images.append(image_t)

        # start from the first frame
        current_frame_index = 0
        frame_ids = list(range(video.num_annotated_frames))

        while True:
            print(f"Showing frame {frame_ids[current_frame_index] + 1}/{video.num_annotated_frames}")
            cv2.imshow("Video Frame", annotated_images[current_frame_index])

            charin = cv2.waitKey(0)

            if charin == 110:  # 'n'
                break

            elif charin == 113:  # 'q'
                return

            elif charin in (81, 97):  # 'a'
                current_frame_index = max(0, current_frame_index - 1)

            elif charin in (83, 100):  # 'd'
                current_frame_index = min(len(frame_ids) - 1, current_frame_index + 1)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--images_base_dir", required=True)
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--first_frame_annotations", action='store_true')

    main(parser.parse_args())
