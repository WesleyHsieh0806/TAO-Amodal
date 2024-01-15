from typing import Optional, List, Tuple, Union, Dict, Any

import cv2
import numpy as np
import os.path as osp

import burstapi.utils as utils


class BURSTVideo:
    def __init__(self, video_dict: Dict[str, Any], images_dir: Optional[str] = None):

        self.annotated_image_paths: List[str] = video_dict["annotated_image_paths"]
        self.all_images_paths: List[str] = video_dict["all_image_paths"]
        self.segmentations: List[Dict[int, Dict[str, Any]]] = video_dict["segmentations"]
        self._track_category_ids: Dict[int, int] = video_dict["track_category_ids"]
        self.image_size: Tuple[int, int] = (video_dict["height"], video_dict["width"])

        self.id = video_dict["id"]
        self.dataset = video_dict["dataset"]  # e.g., ArgoVerse
        self.name = video_dict["seq_name"]  # e.g., f1008c18-e76e-3c24-adcc-da9858fac145
        self.negative_category_ids = video_dict["neg_category_ids"]
        self.not_exhaustive_category_ids = video_dict["not_exhaustive_category_ids"]

        self._images_dir = images_dir
        self._image_to_mask = {image_name: mask for image_name, mask in zip(self.annotated_image_paths, self.load_masks())}
    
    def is_mask_annotated(self, name):
        ''' 
        * check whether a frame is annotated 
        * Example of the frame name: ring_front_center_315973412018496864.jpg
        '''
        return osp.split(str(name))[-1] in self._image_to_mask

    def get_mask_by_frame(self, name):
        '''
        * Get the mask dictionary by providing the name of the frame
        * Output: mask dictionary with the format: {
                track_id: binary mask (np.ndarray)
            }
        '''
        return self._image_to_mask[osp.split(str(name))[-1]]

    @property
    def num_annotated_frames(self) -> int:
        return len(self.annotated_image_paths)

    @property
    def num_total_frames(self) -> int:
        return len(self.all_images_paths)

    @property
    def image_height(self) -> int:
        return self.image_size[0]

    @property
    def image_width(self) -> int:
        return self.image_size[1]

    @property
    def track_ids(self) -> List[int]:
        return list(sorted(self._track_category_ids.keys()))

    @property
    def track_category_ids(self) -> Dict[int, int]:
        return {
            track_id: self._track_category_ids[track_id]
            for track_id in self.track_ids
        }

    def load_images(self, frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Load annotated image frames for the video
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of images as numpy arrays of dtype uint8 and shape [H, W, 3] (RGB)
        """
        assert self._images_dir is not None, f"Images cannot be loaded because 'images_dir' is None"

        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                f"invalid"

        images = []

        for t in frame_indices:
            filepath = osp.join(self._images_dir, self.annotated_image_paths[t])
            assert osp.exists(filepath), f"Image file not found: '{filepath}'"
            images.append(cv2.imread(filepath, cv2.IMREAD_COLOR)[:, :, ::-1])  # convert BGR to RGB

        return images

    def load_masks(self, frame_indices: Optional[List[int]] = None) -> List[Dict[int, np.ndarray]]:
        """
        Decode RLE masks into mask images
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of dicts (one per frame). Each dict has track IDs as keys and mask images as values.
        """
        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                f"invalid"

        zero_mask = np.zeros(self.image_size, bool)
        masks = []

        for t in frame_indices:
            masks_t = dict()

            for track_id in self.track_ids:
                if track_id in self.segmentations[t]:
                    masks_t[track_id] = utils.rle_ann_to_mask(self.segmentations[t][track_id]["rle"], self.image_size)
                else:
                    masks_t[track_id] = zero_mask

            masks.append(masks_t)

        return masks

    def filter_category_ids(self, category_ids_to_keep: List[int]):
        track_ids_to_keep = [
            track_id for track_id, category_id in self._track_category_ids.items()
            if category_id in category_ids_to_keep
        ]

        self._track_category_ids = {
            track_id: category_id for track_id, category_id in self._track_category_ids.items()
            if track_id in track_ids_to_keep
        }

        for t in range(self.num_annotated_frames):
            self.segmentations[t] = {
                track_id: seg for track_id, seg in self.segmentations[t].items()
                if track_id in track_ids_to_keep
            }

    def stats(self) -> Dict[str, Any]:
        total_masks = 0
        for segs_t in self.segmentations:
            total_masks += len(segs_t)

        return {
            "Annotated frames": self.num_annotated_frames,
            "Object tracks": len(self.track_ids),
            "Object masks": total_masks,
            "Unique category IDs": list(set(self.track_category_ids.values()))
        }

    def load_first_frame_annotations(self) -> List[Dict[int, Dict[str, Any]]]:
        annotations = []
        for t in range(self.num_annotated_frames):
            annotations_t = dict()

            for track_id, annotation in self.segmentations[t].items():
                annotations_t[track_id] = {
                    "mask": utils.rle_ann_to_mask(annotation["rle"], self.image_size),
                    "bbox": annotation["bbox"],  # xywh format
                    "point": annotation["point"]
                }

            annotations.append(annotations_t)

        return annotations
