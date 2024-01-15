from typing import Optional, List, Tuple, Union, Dict, Any
from burstapi.video import BURSTVideo

import json
import os.path as osp

import burstapi.utils as utils


class BURSTDataset:
    def __init__(self, annotations_file: str, images_base_dir: Optional[str] = None):
        with open(annotations_file, 'r') as fh:
            content = json.load(fh)

        # convert track IDs from str to int wherever they are used as dict keys (JSON format always parses dict keys as
        # strings)
        self._videos = [utils.intify_track_ids(video) for video in content["sequences"]]

        self.category_names = {
            category["id"]: category["name"] for category in content["categories"]
        }

        self._split = content["split"]

        self.images_base_dir = images_base_dir

        # map video name to idx
        self._name_to_idx = {}
        for i, video in enumerate(self._videos):
            self._name_to_idx[osp.join(self._split, video["dataset"], video["seq_name"])] = i
            self._name_to_idx[osp.join(video["dataset"], video["seq_name"])] = i
    @property
    def num_videos(self) -> int:
        return len(self._videos)
    
    def get_video_by_name(self, name) -> BURSTVideo:
        '''
        * The name of the video should has the format: [split]/[Dataset]/[Video Name]
            e.g., train/Charades/AVSN8
        '''
        return self.__getitem__(self._name_to_idx[name])

    def __getitem__(self, index) -> BURSTVideo:
        assert index < self.num_videos, f"Index {index} invalid since total number of videos is {self.num_videos}"

        video_dict = self._videos[index]
        if self.images_base_dir is None:
            video_images_dir = None
        else:
            video_images_dir = osp.join(self.images_base_dir, self._split, video_dict["dataset"], video_dict["seq_name"])
            assert osp.exists(video_images_dir), f"Images directory for video not found at expected path: '{video_images_dir}'"

        return BURSTVideo(video_dict, video_images_dir)

    def __iter__(self):
        for i in range(self.num_videos):
            yield self[i]
