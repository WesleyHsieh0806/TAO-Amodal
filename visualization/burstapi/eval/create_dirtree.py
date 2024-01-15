from argparse import ArgumentParser
from typing import List

import asyncio
import os
import os.path as osp
import subprocess
import shutil
import tempfile


def create_temp_directories(pred_path: str, gt_dir_path: str) -> str:
    base_dir = osp.join(tempfile.gettempdir(), "BURST_eval")
    print(f"Creating temporary directory tree at: {base_dir}")

    for split in ("common_classes", "all_classes", "uncommon_classes"):
        out_dir = osp.join(base_dir, "gt", "val_or_test", split)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(osp.join(gt_dir_path, f"{split}.json"), osp.join(out_dir, f"{split}.json"))

    pred_out_dir = osp.join(base_dir, "trackers", "Tracker1", "data")
    os.makedirs(pred_out_dir, exist_ok=True)
    shutil.copyfile(pred_path, osp.join(pred_out_dir, "results.json"))

    return base_dir


def main(args):
    assert osp.isdir(args.gt), f"'--gt' must point to a directory"
    for required_gt_file in ("all_classes.json", "common_classes.json", "uncommon_classes.json"):
        expected_path = osp.join(args.gt, required_gt_file)
        if not osp.exists(expected_path):
            raise FileNotFoundError(f"Following ground-truth file not found: {expected_path}")

    assert osp.isfile(args.pred), f"'--pred' must point a file"

    trackeval_dir = os.environ.get("TRACKEVAL_DIR", None)
    if trackeval_dir is None:
        raise EnvironmentError(f"Environment variable 'TRACKEVAL_DIR' is not set.")

    trackeval_scripts_dir = osp.join(trackeval_dir, "scripts")
    if not osp.exists(trackeval_scripts_dir):
        raise FileNotFoundError(f"'scripts' directory not found under $TRACKEVAL_DIR")

    output_bash_file_path = osp.join(trackeval_scripts_dir, "_tmp_burst_eval.sh")
    if osp.exists(output_bash_file_path):
        os.remove(output_bash_file_path)

    base_dir = create_temp_directories(args.pred, args.gt)
    trackers_folder = osp.join(base_dir, "trackers")

    if args.task == "class_guided":
        eval_script = "run_burst.py"
        if not osp.exists(osp.join(trackeval_scripts_dir, eval_script)):
            raise FileNotFoundError(f"Could not find eval script at: {osp.join(trackeval_scripts_dir, eval_script)}")

        command = [
            "python", eval_script, "--USE_PARALLEL", "True", "--EXEMPLAR_GUIDED", "False", "--GT_FOLDER",
            osp.join(base_dir, "gt", "val_or_test", "all_classes"), "--TRACKERS_FOLDER", trackers_folder
        ]
        command = " ".join(command)

    elif args.task == "exemplar_guided":
        eval_script = "run_burst.py"
        if not osp.exists(osp.join(trackeval_scripts_dir, eval_script)):
            raise FileNotFoundError(
                f"Could not find eval script at: {osp.join(trackeval_scripts_dir, eval_script)}")

        command = [
            "python", eval_script, "--USE_PARALLEL", "True", "--EXEMPLAR_GUIDED", "True", "--GT_FOLDER",
            osp.join(base_dir, "gt", "val_or_test", "all_classes"), "--TRACKERS_FOLDER", trackers_folder
        ]
        command = " ".join(command)

    elif args.task == "open_world":
        eval_script = "run_burst_ow.py"

        command = []
        for class_split in ("all_classes", "common_classes", "uncommon_classes"):
            gt_folder = osp.join(base_dir, "gt", "val_or_test", class_split)
            command.append(f"echo \"EVALUATING RESULT FOR {class_split}\"")
            command.append(f"python {eval_script} --USE_PARALLEL True --GT_FOLDER {gt_folder} "
                           f"--TRACKERS_FOLDER {trackers_folder}")

        command = "\n".join(command)

    else:
        raise ValueError("Should not be here.")

    with open(output_bash_file_path, 'w') as fh:
        fh.write(f"{command}\n")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--pred", required=True,
                        help="Path to predicted results JSON file.")

    parser.add_argument("--gt", required=True,
                        help="path to directory containing the three ground-truth files: 'common_classes.json', "
                             "'all_classes.json' and 'uncommon_classes.json'")

    parser.add_argument("--task", required=True, choices=("class_guided", "open_world", "exemplar_guided"),
                        help="For common and long-tail, set this to 'class_guided'. Other two choices are "
                             "self-explanatory.")

    main(parser.parse_args())
