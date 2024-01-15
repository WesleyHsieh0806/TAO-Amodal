from typing import Dict, List, Any, Tuple

import numpy as np
import pycocotools.mask as cocomask


def intify_track_ids(video_dict: Dict[str, Any]):
    video_dict["track_category_ids"] = {
        int(track_id): category_id for track_id, category_id in video_dict["track_category_ids"].items()
    }

    for t in range(len(video_dict["segmentations"])):
        video_dict["segmentations"][t] = {
            int(track_id): seg
            for track_id, seg in video_dict["segmentations"][t].items()
        }

    return video_dict


def rle_ann_to_mask(rle: str, image_size: Tuple[int, int]) -> np.ndarray:
    return cocomask.decode({
        "size": image_size,
        "counts": rle.encode("utf-8")
    }).astype(bool)


def mask_to_rle_ann(mask: np.ndarray) -> Dict[str, Any]:
    assert mask.ndim == 2, f"Mask must be a 2-D array, but got array of shape {mask.shape}"
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
