import itertools
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple

from tqdm import tqdm


CACHE_DIR = Path(__file__).resolve().parent.parent / 'cache/imagenetvid'
CACHE_DIR.mkdir(exist_ok=True, parents=True)


CLASSES = {
   "0": ["n02691156", "airplane"],
   "1": ["n02419796", "antelope"],
   "2": ["n02131653", "bear"],
   "3": ["n02834778", "bicycle"],
   "4": ["n01503061", "bird"],
   "5": ["n02924116", "bus"],
   "6": ["n02958343", "car"],
   "7": ["n02402425", "cattle"],
   "8": ["n02084071", "dog"],
   "9": ["n02121808", "domestic_cat"],
   "10": ["n02503517", "elephant"],
   "11": ["n02118333", "fox"],
   "12": ["n02510455", "giant_panda"],
   "13": ["n02342885", "hamster"],
   "14": ["n02374451", "horse"],
   "15": ["n02129165", "lion"],
   "16": ["n01674464", "lizard"],
   "17": ["n02484322", "monkey"],
   "18": ["n03790512", "motorcycle"],
   "19": ["n02324045", "rabbit"],
   "20": ["n02509815", "red_panda"],
   "21": ["n02411705", "sheep"],
   "22": ["n01726692", "snake"],
   "23": ["n02355227", "squirrel"],
   "24": ["n02129604", "tiger"],
   "25": ["n04468005", "train"],
   "26": ["n01662784", "turtle"],
   "27": ["n04530566", "watercraft"],
   "28": ["n02062744", "whale"],
   "29": ["n02391049", "zebra"],
   "30": ["n00001740", "entity"]
}


class Box(NamedTuple):
    x_min: (float)
    y_min: (float)
    x_max: (float)
    y_max: (float)

    @staticmethod
    def from_xml(label_xml):
        if isinstance(label_xml, str):
            label_xml = ET.fromstring(label_xml)
        return Box(
            x_min=int(label_xml.find('xmin').text),
            y_min=int(label_xml.find('ymin').text),
            x_max=int(label_xml.find('xmax').text),
            y_max=int(label_xml.find('ymax').text))


class ImagenetVidObjectLabel(NamedTuple):
    track_id: int
    wordnet_id: str
    box: Box
    occluded: bool  # TODO: Figure out what this means.
    generated: bool  # TODO: Figure out what this means.

    @staticmethod
    def from_xml(label_xml):
        if isinstance(label_xml, str):
            label_xml = ET.fromstring(label_xml)
        assert label_xml.find('occluded').text in ('0', '1')
        assert label_xml.find('generated').text in ('0', '1')
        return ImagenetVidObjectLabel(
            track_id=int(label_xml.find('trackid').text),
            wordnet_id=label_xml.find('name').text,
            box=Box.from_xml(label_xml.find('bndbox')),
            occluded=label_xml.find('occluded').text == '1',
            generated=label_xml.find('generated').text == '1')


class ImagenetVidLabel(NamedTuple):
    sequence: str
    filename: str
    height: int
    width: int
    objects: List[ImagenetVidObjectLabel]

    def get_labels(self):
        return list(set(x.wordnet_id for x in self.objects))

    @staticmethod
    def from_xml(label_xml):
        if isinstance(label_xml, str):
            label_xml = ET.fromstring(label_xml)
        # Example label:
        #   <folder>
        #       ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000
        #   </folder>
        #   <filename>000032</filename>
        #   <source>
        #           <database>ILSVRC_2015</database>
        #   </source>
        #   <size>
        #           <width>1280</width>
        #           <height>720</height>
        #   </size>
        #   <object>
        #           <trackid>0</trackid>
        #           <name>n01674464</name>
        #           <bndbox>
        #                   <xmax>1042</xmax>
        #                   <xmin>298</xmin>
        #                   <ymax>420</ymax>
        #                   <ymin>219</ymin>
        #           </bndbox>
        #           <occluded>1</occluded>
        #           <generated>1</generated>
        #   </object>
        sequence = os.path.split(label_xml.find('folder').text)[-1]
        filename = label_xml.find('filename').text
        height = int(label_xml.find('./size/height').text)
        width = int(label_xml.find('./size/width').text)
        objects = [
            ImagenetVidObjectLabel.from_xml(object_xml)
            for object_xml in label_xml.findall('object')
        ]
        return ImagenetVidLabel(
            sequence=sequence,
            filename=filename,
            height=height,
            width=width,
            objects=objects)


def parse_label(xml_str):
    """Helper wrapper for ImagenetVidLabel.from_xml."""
    if isinstance(xml_str, bytes):
        xml_str = xml_str.decode()
    return ImagenetVidLabel.from_xml(xml_str)


def to_coco_json(imagenetvid_root, include_2017_labels=True):
    imagenetvid_root = Path(imagenetvid_root)
    cache_paths = {
        'train': CACHE_DIR / 'train.json',
        'val': CACHE_DIR / 'val.json'
    }
    if not include_2017_labels:
        cache_paths['train'] = CACHE_DIR / 'train_2015_only.json'
        cache_paths['val'] = CACHE_DIR / 'val_2015_only.json'

    if cache_paths['train'].exists() and cache_paths['val'].exists():
        info = {}
        with open(cache_paths['train'], 'r') as f:
            info['train'] = json.load(f)
        with open(cache_paths['val'], 'r') as f:
            info['val'] = json.load(f)
        return info

    annotations_dir = imagenetvid_root / 'Annotations' / 'VID'
    assert annotations_dir.exists()

    date = datetime.now()
    dataset_info = {
        'year': date.year,
        'version': '1.0',
        'description': 'ImageNet VID dataset in COCO format.',
        'contributor': ('Dataset from Olga Russakovsky, Jia Deng, Hao Su, '
                        'Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng '
                        'Huang, Andrej Karpathy, Aditya Khosla, Michael '
                        'Bernstein, Alexander C. Berg and Li Fei-Fei.'
                        'COCO format by Achal Dave.'),
        'url': 'http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php',
        'date_created': str(date)
    }
    long_license = 'Unknown'

    categories = []
    wordnet_id_to_index = {}
    for label_index, (wordnet_id, name) in CLASSES.items():
        categories.append({
            'supercategory': 'object',
            'id': int(label_index) + 1,
            'name': name
        })
        wordnet_id_to_index[wordnet_id] = int(label_index) + 1
    output_info = {}
    for split in ['train', 'val']:
        output_info[split] = {
            'info': dataset_info,
            'images': [],
            'annotations': [],
            'categories': categories,
            'licenses': [long_license]
        }

    # Load image ids for val set.
    with open(imagenetvid_root / 'ImageSets' / 'VID' / 'val.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]

    # Maps, e.g., 'ILSVRC2015_val_00000000/000001' -> 1
    val_image_ids = {
        image_path: int(image_id)
        for image_path, image_id in lines
    }

    train_image_id_generator = itertools.count(
        start=max(val_image_ids.values()) + 1)
    annotation_id_generator = itertools.count()

    # Annotation format:
    # annotation {
    #     "id" : int,
    #     "image_id" : int,
    #     "category_id" : int,
    #     "segmentation" : RLE or [polygon],
    #     "area" : float,
    #     "bbox" : [x,y,width,height],
    #     "iscrowd" : 0 or 1,
    # }
    #
    # Images format:
    #
    # image{
    #     "id" : int,
    #     "width" : int,
    #     "height" : int,
    #     "file_name" : str,
    #     "license" : int,
    #     "flickr_url" : str,
    #     "coco_url" : str,
    #     "date_captured" : datetime,
    # }

    for split in ('train', 'val'):
        if cache_paths[split].exists():
            with open(cache_paths[split], 'r') as f:
                output_info[split] = json.load(f)
            continue
        split_dir = annotations_dir / split
        assert split_dir.exists(), ('Missing directory: %s' % split_dir)

        # Explicitly handle one layer of symlinks
        files = []
        for x in tqdm(list(split_dir.iterdir()),
                      desc=f'Collecting {split} xmls'):
            files.extend(x.rglob('*.xml'))
        # files = list(tqdm(file_generator, desc=f'Collecting {split} paths'))
        files = sorted(files, key=lambda x: (x.parent.stem, int(x.stem)))
        if not include_2017_labels:
            total_files = len(files)
            files = [x for x in files if 'ILSVRC2017' not in x.parent.stem]
            logging.info(f'Removed {total_files-len(files)}/{total_files} '
                         f'belonging to 2017 dataset.')
        for xml_path in tqdm(files, desc=f'Parsing {split} XMLs'):
            if (not include_2017_labels
                    and 'ILSVRC2017' in xml_path.parent.stem):
                continue
            with open(xml_path, 'r') as f:
                data = parse_label(f.read())
            if split == 'val':
                image_id = val_image_ids[
                    f'{xml_path.parent.name}/{xml_path.stem}']
            else:
                image_id = next(train_image_id_generator)
            jpeg_path = (
                xml_path.relative_to(annotations_dir).with_suffix('.JPEG'))
            output_info[split]['images'].append({
                'id': image_id,
                'width': data.width,
                'height': data.height,
                'file_name': str(jpeg_path),
                'coco_url': '',
                'date_captured': ''
            })
            for label in data.objects:
                x0, y0, x1, y1 = (label.box.x_min, label.box.y_min,
                                  label.box.x_max, label.box.y_max)
                is_generated = int(label.generated) == 1
                # To 0-indexed coordinates
                x0 -= 1
                y0 -= 1
                x1 -= 1
                y1 -= 1
                # Need to add 1 as COCO annotations assume width is
                # non-inclusive (I think; I'm mostly copying this from
                # https://github.com/cocodataset/cocoapi/blob/aca78bcd6b4345d25405a64fdba1120dfa5da1ab/MatlabAPI/CocoUtils.m#L336
                # )
                w, h = x1 - x0 + 1, y1 - y0 + 1
                output_info[split]['annotations'].append({
                    'id': next(annotation_id_generator),
                    'track_id': label.track_id,
                    'image_id': image_id,
                    'category_id': wordnet_id_to_index[label.wordnet_id],
                    'bbox': [x0, y0, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                    'segmentation': [],
                    'generated': is_generated
                })
        output_path = cache_paths[split]
        logging.info('Outputting %s annotations to %s', split, output_path)
        with open(output_path, 'w') as f:
            json.dump(output_info[split], f)
    return output_info


def load_image_ids(imagenet_set_path):
    with open(imagenet_set_path) as f:
        lines = [x.strip().split(' ') for x in f.readlines()]

    # Maps, e.g., 'ILSVRC2015_val_00000000/000001' -> 1
    return {
        image_path: int(image_id)
        for image_path, image_id in lines
    }


def load_tracks(predictions_txt, progress=False):
    """
    Args:
        predictions_txt (Path)

    Returns:
        tracks (list): List of
            (frame_id, label, track_id, conf, x0, y0, x1, y1)
    """
    # Format: <frame_id> <label> <track_id> <conf> <x0> <y0> <x1> <y1>
    with open(predictions_txt, 'r') as f:
        # Read the line at once to memory.
        lines = f.readlines()

    tracks = []
    for line in tqdm(lines,
                     desc='Parsing predictions',
                     mininterval=1,
                     disable=not progress):
        fields = line.strip().split(' ')
        frame_id = int(fields[0])
        label = int(fields[1])
        track_id = int(fields[2])
        conf = float(fields[3])
        x0 = float(fields[4])
        y0 = float(fields[5])
        x1 = float(fields[6])
        y1 = float(fields[7])
        tracks.append((frame_id, label, track_id, conf, x0, y0, x1, y1))

    return tracks
