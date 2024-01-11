import json
from tqdm import tqdm
import logging
import os
import pickle
from pathlib import Path
from collections import defaultdict
import itertools
import copy
import sys 
import argparse
from typing import Mapping

sys.path.insert(0, '../')

import numpy as np

from tao_amodal.evaluation.tao_amodal import TaoEval, Tao, TaoResults
from tao_amodal.evaluation.lvis_amodal import LVISEval, LVISResults
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import inference_on_dataset, print_csv_format

def default_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Common parameters across different visualization
    parser.add_argument('--track_result', type=str, required=True)
    parser.add_argument('--output_log', type=str, required=True)
    parser.add_argument('--annotation', type=str, default=None)
    

    args = parser.parse_args()
    return args


args = default_arg_parser()
annotation='/compute/trinity-1-38/chengyeh/TAO/amodal_annotations/validation_with_freeform_amodal_boxes_Aug10_2022_oof_visibility_GTR_lvis_v1.json' if not args.annotation else args.annotation
track_result=args.track_result
output_log=Path(args.output_log)


def make_track_ids_unique(result_anns):
    track_id_videos = {}
    track_ids_to_update = set()
    max_track_id = 0
    for ann in tqdm(result_anns):
        t = ann['track_id']
        if t not in track_id_videos:
            track_id_videos[t] = ann['video_id']

        if ann['video_id'] != track_id_videos[t]:
            # Track id is assigned to multiple videos
            track_ids_to_update.add(t)
        max_track_id = max(max_track_id, t)

    if track_ids_to_update:
        next_id = itertools.count(max_track_id + 1)
        new_track_ids = defaultdict(lambda: next(next_id))
        for ann in tqdm(result_anns):
            t = ann['track_id']
            v = ann['video_id']
            if t in track_ids_to_update:
                ann['track_id'] = new_track_ids[t, v]
    return len(track_ids_to_update)

def _custom_evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, logger=None, class_names=None):
    """
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", 
                 "AP-HO", "AP50-HO", "AP75-HO",
                 "AP-PO", "AP50-PO", "AP75-PO",
                 "AP-HV", "AP50-HV", "AP75-HV",
                 "AP-OOF", "AP50-OOF", "AP75-OOF",
                 "AP-HP", "AP50-HP", "AP75-HP", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", 
                 "AP-HO", "AP50-HO", "AP75-HO",
                 "AP-PO", "AP50-PO", "AP75-PO",
                 "AP-HV", "AP50-HV", "AP75-HV",
                 "AP-OOF", "AP50-OOF", "AP75-OOF",
                 "AP-HP", "AP50-HP", "AP75-HP", "APr", "APc", "APf"],
    }[iou_type]

    if logger is None:
        logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        for c in lvis_results:
            c.pop("bbox", None)

    logger.info('Evaluating {} on LVIS...'.format(lvis_results))
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}


    logger.info("Evaluation results for {}: \n".format(
        iou_type) + create_small_table(results))
    del lvis_eval

    important_res = [(metric, results[metric]) for metric in metrics]
    logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
    logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
    return results

def eval_tao_track(ann_path, results_path, logger, class_names=None):
    logger.setLevel(logging.INFO)
    results = {}
    logger.info("Loading gt {}...".format(ann_path))
    tao_gt = Tao(ann_path)
    logger.info('Done')
    logger.info('Loading results...')
    
    # preprocess results
    with open(results_path, 'r') as f:
        tao_results_ann = json.load(f)
    make_track_ids_unique(tao_results_ann)

    logger.info('Done')
    # import pdb; pdb.set_trace()
    logger.info('Building')
    tao_eval = TaoEval(tao_gt, tao_results_ann, logger=logger)
    logger.info('Done')
   
    tao_eval.run()
    tao_eval.print_results()
    results["TAO 3DmAP50"] = tao_eval.get_results()["AP50"] * 100
    results["TAO 3DmAP50-HP"] = tao_eval.get_results()["AP50-HP"] * 100
    results["TAO 3DmAP"] = tao_eval.get_results()["AP"] * 100
    results["TAO 3DmAP-HP"] = tao_eval.get_results()["AP-HP"] * 100
    
    logger.info("TAO 3DmAP50:{:.4f}".format(results["TAO 3DmAP50"]))
    logger.info("TAO 3DmAP50-HP:{:.4f}".format(results["TAO 3DmAP50-HP"]))
    logger.info("TAO 3DmAP:{:.4f}".format(results["TAO 3DmAP"]))
    logger.info("TAO 3DmAP-HP:{:.4f}".format(results["TAO 3DmAP-HP"]))
    keys = ["TAO 3DmAP50", "TAO 3DmAP50-HP", "TAO 3DmAP", "TAO 3DmAP-HP"]
    
    logger.info("copypaste: " + ",".join([k for k in keys]))
    logger.info("copypaste: " + ",".join(["{:.4f}".format(results[k]) for k in keys]))



# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
output_log.parent.mkdir(parents=True, exist_ok=True)
fileHandler = logging.FileHandler(output_log, mode='w')
logger.addHandler(fileHandler)

# evaluate detecion and tracking performance
_custom_evaluate_predictions_on_lvis(annotation, track_result, "bbox", logger=logger)

eval_tao_track(annotation, track_result, logger)