import argparse
import logging
import pickle

from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

from tao.utils import misc


def parse_bool(arg):
    """Parse string to boolean.

    Using type=bool in argparse does not do the right thing. E.g.
    '--bool_flag False' will parse as True. See
    <https://stackoverflow.com/q/15008758/1291812>
    """
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected 'True' or 'False'.")


def load_detection_mat(mat):
    dictionary = {}
    f = loadmat(mat)['x']
    result = {}
    result['pred_boxes'] = [[x[0], x[1], x[2], x[3]] for x in f[:, :4]]
    result['scores'] = [x for x in f[:, 5]]
    result['pred_classes'] = [x for x in f[:, 4]]
    dictionary['instances'] = result
    return dictionary


def load_detection_dir_as_results(root,
                                  annotations,
                                  detections_format='pickle',
                                  include_masks=False,
                                  score_threshold=None,
                                  max_dets_per_image=None,
                                  show_progress=False):
    """Load detections from dir as a results.json dict."""
    if not isinstance(root, Path):
        root = Path(root)
    ext = '.pickle' if detections_format == 'pickle' else '.mat'
    bbox_annotations = []
    if include_masks:
        segmentation_annotations = []

    for image in tqdm(annotations['images'],
                      desc='Collecting annotations',
                      disable=not show_progress):
        path = (root / f'{image["file_name"]}').with_suffix(ext)
        if not path.exists():
            logging.warn(f'Could not find detections for image '
                         f'{image["file_name"]} at {path}; skipping...')
            continue
        if detections_format == 'pickle':
            with open(path, 'r') as f:
                detections = pickle.load(f)
        else:
            detections = misc.load_detection_mat(path)

        num_detections = len(detections['instances']['scores'])
        indices = sorted(range(num_detections),
                         key=lambda i: detections['instances']['scores'][i],
                         reverse=True)

        if max_dets_per_image is not None:
            indices = indices[:max_dets_per_image]

        for idx in indices:
            entry = detections['instances']['pred_boxes'][idx]
            x1 = entry[0]
            y1 = entry[1]
            x2 = entry[2]
            y2 = entry[3]
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]

            category = int(detections['instances']['pred_classes'][idx] + 1)
            score = detections['instances']['scores'][idx]
            if score_threshold is not None and score < score_threshold:
                continue

            try:
                score = score.item()
            except AttributeError:
                pass

            bbox_annotations.append({
                'image_id': image['id'],
                'category_id': category,
                'bbox': bbox,
                'score': score,
            })
            if include_masks:
                segmentation_annotations.append({
                    'image_id': image['id'],
                    'category_id': category,
                    'segmentation': detections['instances']['pred_masks'][idx],
                    'score': score
                })
    if include_masks:
        return bbox_annotations, segmentation_annotations
    else:
        return bbox_annotations
