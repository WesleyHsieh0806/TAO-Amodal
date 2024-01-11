# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou: add tracking evaluation
import copy
import itertools
import json
import logging
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import numpy as np

from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

from lvis import LVISEval, LVISResults
from tao.toolkit.tao import TaoEval, Tao, TaoResults
import pycocotools.mask as mask_util



def custom_instances_to_lvis_json(instances, img_id):
    """
    Add track_id
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_track_id = instances.has("track_ids")
    if has_track_id:
        track_ids = instances.track_ids

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_track_id:
            result['track_id'] = int(track_ids[k].item())
            result['video_id'] = result['track_id'] // 100000000
        results.append(result)
    
    return results


class CustomTAOEvaluator(LVISEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API. (TAO and TAO-Amodal could also be evaluated using this evaluator)
    """

    def __init__(self, dataset_name, tasks, distributed, output_dir=None, not_eval=False, max_dets_per_image=None):
        '''
        * To Customize the metric evaluation process, we could redefine
        *   1. _eval_box_proposals
        *   2. _eval_predictions
        * , since they are called during evaluator.evaluate()
        * https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/lvis_evaluation.html#LVISEvaluator
        '''
        # The task of this evaluator is intiailized using configuration node
        # it's deprecated, so the task is inferred automatically using _tasks_from_predictions
        super().__init__(dataset_name, tasks, distributed, output_dir=output_dir, max_dets_per_image=max_dets_per_image)
        self._do_evaluation = not (not_eval)


    def _tasks_from_predictions(self, predictions):
        for pred in predictions:
            if 'track_id' in pred and 'segmentation' in pred:
                return ("bbox", "segm", '_track')
            if 'track_id' in pred:
                return ("bbox", '_track')
            if "segmentation" in pred:
                return ("bbox", "segm")
        return ("bbox",)

    def process(self, inputs, outputs):
        """
        support dump feature
        inputs: a list of dict, where each dict corresponds to an image and 
            contains keys like “height”, “width”, “file_name”, “image_id”.
        Outputs:
            It is a list of dicts with key “instances” that contains Instances.
        Refer to https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format for the input/output format
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = custom_instances_to_lvis_json(
                    instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)


    def _eval_predictions(self, predictions, img_ids=None):
        """
        """
        assert img_ids is None
        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            if task == '_track':
                json_file = PathManager.get_local_path(self._metadata.json_file)
                file_path = os.path.join(
                    self._output_dir, "lvis_instances_results.json")
                res = _eval_tao_track(
                    json_file, file_path, self._logger,
                    class_names=self._metadata.get("thing_classes"))
                self._results[task] = res
            else:
                res = _custom_evaluate_predictions_on_lvis(
                    self._lvis_api, lvis_results, task, 
                    logger=self._logger,
                    class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        # areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        # limits = [100, 300, 1000]
        areas = {"all": ""}
        limits = [300]
        for limit in limits:
            for area, suffix in areas.items():
                stats = _custom_evaluate_box_proposals(
                    predictions, self._lvis_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
                key = "AR50{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar50"].item() * 100)

        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _custom_evaluate_box_proposals(
    dataset_predictions, lvis_api, area="all", limit=None, cat=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official LVIS API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0
    cat_ids = None if cat is None else [cat]

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = lvis_api.get_ann_ids(img_ids=[prediction_dict["image_id"]], cat_ids=cat_ids)
        anno = lvis_api.load_anns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in anno
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    if num_pos == 0:
        return {'ar': torch.tensor(-1.), 'ar50': 0, 'num_pos': 0}
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    step = 0.05
    thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    ar50 = recalls[0]
    return {
        "ar": ar,
        "ar50": ar50,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _custom_evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, logger=None, class_names=None):
    """
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
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

    lvis_results = LVISResults(lvis_gt, lvis_results)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}

    precisions = lvis_eval.eval['precision']
    assert len(class_names) == precisions.shape[2]
    results_per_category = []
    id2apiid = sorted(lvis_gt.get_cat_ids())
    inst_aware_ap, inst_count = 0, 0
    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        inst_num = len(lvis_gt.get_ann_ids(cat_ids=[id2apiid[idx]]))
        if inst_num > 0:
            results_per_category.append(("{} {}".format(
                name, 
                inst_num if inst_num < 1000 else '{:.1f}k'.format(inst_num / 1000)), 
                float(ap * 100)))
            inst_aware_ap += inst_num * ap
            inst_count += inst_num
    inst_aware_ap = inst_aware_ap * 100 / inst_count

    # tabulate it
    # N_COLS = min(6, len(results_per_category) * 2)
    # results_flatten = list(itertools.chain(*results_per_category))
    # results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    # table = tabulate(
    #     results_2d,
    #     tablefmt="pipe",
    #     floatfmt=".3f",
    #     headers=["category", "AP"] * (N_COLS // 2),
    #     numalign="left",
    # )
    # logger.info("Per-category {} AP: \n".format(iou_type) + table)

    logger.info("Evaluation results for {}: \n".format(
        iou_type) + create_small_table(results))
    del lvis_eval
    return results

def _eval_tao_track(ann_path, results_path, logger, class_names=None):
    logger.setLevel(logging.INFO)
    results = {}
    print("Loading gt...")

    '''
    * Refer to https://github.com/TAO-Dataset/tao/blob/master/tao/toolkit/tao/tao.py
    * to see how TAO toolkit load annotation files
    '''
    tao_gt = Tao(ann_path)
    print('Done')
    print('Loading results...')
    tao_results = TaoResults(tao_gt, results_path)
    print('Done')
    # import pdb; pdb.set_trace()
    print('Building')
    tao_eval = TaoEval(tao_gt, tao_results, logger=logger)
    print('Done')
   
    tao_eval.run()
    tao_eval.print_results()
    results["TAO 3DmAP"] = tao_eval.get_results()["AP"] * 100

    precisions = tao_eval.eval['precision']
    assert len(class_names) == precisions.shape[2]
    results_per_category = []
    id2apiid = sorted(tao_gt.get_cat_ids())
    inst_aware_ap, inst_count = 0, 0
    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        inst_num = len(tao_gt.get_ann_ids(cat_ids=[id2apiid[idx]]))
        if inst_num > 0:
            results_per_category.append(("{} {}".format(
                name, 
                inst_num if inst_num < 1000 else '{:.1f}k'.format(inst_num / 1000)), 
                float(ap * 100)))
            inst_aware_ap += inst_num * ap
            inst_count += inst_num
    inst_aware_ap = inst_aware_ap * 100 / inst_count
    # tabulate it
    # N_COLS = min(6, len(results_per_category) * 2)
    # results_flatten = list(itertools.chain(*results_per_category))
    # results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    # table = tabulate(
    #     results_2d,
    #     tablefmt="pipe",
    #     floatfmt=".3f",
    #     headers=["category", "3d AP"] * (N_COLS // 2),
    #     numalign="left",
    # )
    # iou_type = '3D'
    # logger.info("Per-category {} AP: \n".format(iou_type) + table)
    return results
