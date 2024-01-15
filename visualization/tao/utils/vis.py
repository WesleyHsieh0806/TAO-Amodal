import cv2

import numpy as np
import pycocotools.mask as mask_util

from tao.utils import cv2_util
from tao.utils.colormap import colormap


_BLACK = (0, 0, 0)
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_RED = (255, 18, 15)
_BLUE = (18, 18, 255)
_WHITE = (255, 255, 255)

_COLOR1 = tuple(255*x for x in (0.000, 0.447, 0.741))


def get_annotation_colors(annotations):
    # Sort boxes by area, this will ensure, e.g., that the largest box
    # has the same color in all frames of a video.
    areas = [x['bbox'][2] * x['bbox'][3] for x in annotations]
    box_order = sorted(range(len(areas)), key=lambda i: areas[i])
    colors = colormap(rgb=True)[:len(annotations)].tolist()
    return [colors[i % len(colors)] for i in box_order]

def vis_visibility(image,
              pos,
              visibility_str,
              font_scale=0.1,
              bg_color=_BLACK,
              text_color=_GRAY,
              thickness=1):
    """Visualizes the class."""
    x, y = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = visibility_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)

    # Place text background.
    back_tl = x, y - int(1.1 * txt_h)
    back_br = x + txt_w, y

    # Show text.
    # txt_tl = x, y + int(1 * txt_h)
    txt_tl = x, y 
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)
    cv2.putText(image,
                txt,
                txt_tl,
                font,
                font_scale,
                text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
    return image

def overlay_amodal_visibility_coco(image,
                       annotations,
                       categories,
                       oy=None,
                        ox=None,
                       background_colors=None,
                       font_scale=0.1,
                       font_thickness=1,
                       text_color=_GRAY,
                       show_track_id=False):
    """
    Adds class names in the positions defined by the top-left corner of the
    COCO annotation bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
        categories (dict): coco.cats
    """
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    assert not isinstance(categories, list), (
        'categories should be a dict with category ids as keys.')
    labels = []

    ''' 
    * Create text for visibility (IoU between modal box and amodal boxes)
    '''
    for a in annotations:
        amodal_box = a["amodal_bbox"]
        if "bbox" in a:
            bbox = a["bbox"]
            label = "Vis:" + " " + "{:.3f}".format(bb_intersection_over_union([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], 
                        [amodal_box[0], amodal_box[1], amodal_box[0] + amodal_box[2], amodal_box[1] + amodal_box[3]]))
            label = label + " " + ("X" if a["amodal_is_uncertain"] else "O")
        elif "segmentation" in a:
            segmentation = a["segmentation"][0]
            label = "Vis:" + " " + "{:.3f}".format(bb_intersection_over_union([segmentation[0], segmentation[1], segmentation[4], segmentation[5]], 
                        [amodal_box[0], amodal_box[1], amodal_box[0] + amodal_box[2], amodal_box[1] + amodal_box[3]]))
            label = label + " " + ("X" if a["amodal_is_uncertain"] else "O")
        else:
            label = ("Vis:0.000 X" if a["amodal_is_uncertain"] else "Vis:0.000 O")

        labels.append(label)
    
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    
    boxes = [[int(round(y)) for y in x['amodal_bbox']] for x in annotations]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    if background_colors is None:
        # colors = get_annotation_colors(annotations)
        colors = [_BLACK for _ in annotations]
    else:
        colors = background_colors

    '''
    * Create Text for each Bounding Box
    '''
    for box, label, color in zip(boxes, labels, colors):
        vis_visibility(image,
                  box,
                  label,
                  font_scale=font_scale,
                  bg_color=color,
                  text_color=text_color,
                  thickness=font_thickness)

    return image

def vis_class(image,
              pos,
              class_str,
              font_scale=0.35,
              bg_color=_WHITE,
              text_color=_BLACK,
              box_thickness=3,
              thickness=1):
    """Visualizes the class."""
    x, y = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_TRIPLEX 
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x, y - int(1.7 * txt_h)
    back_br = int(x + 1.1 * txt_w), y - int(0.2 * txt_h)
    # Show text.
    txt_tl = int(x + 0.05 * txt_w), y - int(0.6 * txt_h)
    cv2.rectangle(image, back_tl, back_br, _WHITE, -1)
    cv2.rectangle(image, back_tl, back_br, bg_color, box_thickness)
    cv2.putText(image,
                txt,
                txt_tl,
                font,
                font_scale,
                text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
    return image


def overlay_amodal_class_coco(image,
                       annotations,
                       categories,
                       oy=None, ox=None,
                       background_colors=None,
                       font_scale=0.5,
                       font_thickness=1,
                       text_color=_BLACK,
                       show_track_id=False):
    """
    Adds class names in the positions defined by the top-left corner of the
    COCO annotation bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
        categories (dict): coco.cats
    """
    assert not isinstance(categories, list), (
        'categories should be a dict with category ids as keys.')
    labels = []
    for a in annotations:
        label = categories[a['category_id']]['name']
        if label == 'baby':
            label = 'person'
        # elif label == 'car_(automobile)':
        #     label = 'car'
        if show_track_id and 'track_id' in a:
            label = f'{label} ({a["track_id"]})'
        labels.append(label)
    # labels = [categories[i['category_id']]['name'] for i in annotations]
    # labels = predictions.get_field("labels").tolist()
    # labels = [categories[i] for i in labels]
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    boxes = [[int(round(y)) for y in x['amodal_bbox']] for x in annotations]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    if background_colors is None:
        # colors = get_annotation_colors(annotations)
        colors = [_WHITE for _ in annotations]
    else:
        colors = background_colors

    for box, label, color in zip(boxes, labels, colors):
        vis_class(image,
                  box,
                  label,
                  font_scale=font_scale,
                  bg_color=color,
                  text_color=text_color,
                  thickness=font_thickness)

    return image

def overlay_amodal_class_prediction(image,
                       annotations,
                       categories,
                       oy=None, ox=None,
                       background_colors=None,
                       font_scale=0.5,
                       font_thickness=1,
                       text_color=_BLACK,
                       show_track_id=False):
    """
    Adds class names in the positions defined by the top-left corner of the
    COCO annotation bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
        categories (dict): coco.cats
    """
    assert not isinstance(categories, list), (
        'categories should be a dict with category ids as keys.')
    labels = []
    for a in annotations:
        label = categories[a['category_id']]['name']
        if label == 'baby':
            label = 'person'
        if show_track_id and 'track_id' in a:
            label = f'{label} ({a["track_id"]})'
        labels.append(label)
    # labels = [categories[i['category_id']]['name'] for i in annotations]
    # labels = predictions.get_field("labels").tolist()
    # labels = [categories[i] for i in labels]
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    boxes = [[int(round(y)) for y in x['bbox']] for x in annotations]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    if background_colors is None:
        # colors = get_annotation_colors(annotations)
        colors = [_BLACK for _ in annotations]
    else:
        colors = background_colors

    for box, label, color in zip(boxes, labels, colors):
        vis_class(image,
                  box,
                  label,
                  font_scale=font_scale,
                  bg_color=color,
                  text_color=text_color,
                  thickness=font_thickness)

    return image

def overlay_class_coco(image,
                       annotations,
                       categories,
                       background_colors=None,
                       font_scale=0.5,
                       font_thickness=1,
                       text_color=_GRAY,
                       show_track_id=False):
    """
    Adds class names in the positions defined by the top-left corner of the
    COCO annotation bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
        categories (dict): coco.cats
    """
    assert not isinstance(categories, list), (
        'categories should be a dict with category ids as keys.')
    labels = []
    for a in annotations:
        label = categories[a['category_id']]['name']
        if label == 'baby':
            label = 'person'
        if show_track_id and 'track_id' in a:
            label = f'{label} ({a["track_id"]})'
        labels.append(label)
    # labels = [categories[i['category_id']]['name'] for i in annotations]
    # labels = predictions.get_field("labels").tolist()
    # labels = [categories[i] for i in labels]
    boxes = [[int(round(y)) for y in x['bbox']] for x in annotations]
    if background_colors is None:
        # colors = get_annotation_colors(annotations)
        colors = [_BLACK for _ in annotations]
    else:
        colors = background_colors

    for box, label, color in zip(boxes, labels, colors):
        vis_class(image,
                  box,
                  label,
                  font_scale=font_scale,
                  bg_color=color,
                  text_color=text_color,
                  thickness=font_thickness)

    return image


def vis_bbox(image,
             box,
             border_color=_BLACK,
             fill_color=_COLOR1,
             fill_opacity=0.65,
             thickness=1):
    """Visualizes a bounding box."""
    x0, y0, w, h = box
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    # Draw border
    if fill_opacity > 0 and fill_color is not None:
        with_fill = image.copy()
        with_fill = cv2.rectangle(with_fill, (x0, y0), (x1, y1),
                                  tuple(fill_color), cv2.FILLED)
        image = cv2.addWeighted(with_fill, fill_opacity, image,
                                1 - fill_opacity, 0, image)
        
    image = cv2.rectangle(image, (x0, y0), (x1, y1), tuple(border_color),
                          thickness)
    return image


def overlay_amodal_boxes_coco(image,
                       annotations, oy=None, ox=None,
                       colors=None,
                       border_color=None,
                       fill_opacity=None,
                       thickness=1):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
    """
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    boxes = [[int(round(y)) for y in x['amodal_bbox']] for x in annotations]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    # track_ids = [x['track_id'] for x in annotations]

    sorted_inds = sorted(range(len(boxes)),
                         key=lambda i: boxes[i][2] * boxes[i][3],
                         reverse=True)

    if colors is None:
        colors = get_annotation_colors(annotations)

    for i in sorted_inds:
        box = boxes[i]
        color = colors[i]
        # track_id = track_ids[i]
        # if track_id != 9:
        #     continue
        kwargs = {}
        if fill_opacity:
            kwargs['fill_opacity'] = fill_opacity
        if border_color is not None:
            kwargs['border_color'] = border_color
        image = vis_bbox(image,
                         box,
                         fill_color=color,
                         border_color=color,
                         thickness=thickness,
                         **kwargs)
    return image

def overlay_modal_boxes_coco(image,
                       annotations, oy=None, ox=None,
                       colors=None,
                       border_color=None,
                       fill_opacity=None,
                       thickness=1):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
    """
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    boxes = [[int(round(y)) for y in x['bbox']] for x in annotations if 'bbox' in x]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    # track_ids = [x['track_id'] for x in annotations]

    sorted_inds = sorted(range(len(boxes)),
                         key=lambda i: boxes[i][2] * boxes[i][3],
                         reverse=True)

    if colors is None:
        colors = get_annotation_colors(annotations)

    for i in sorted_inds:
        box = boxes[i]
        color = colors[i]
        # track_id = track_ids[i]
        # if track_id != 9:
        #     continue
        kwargs = {}
        if fill_opacity:
            kwargs['fill_opacity'] = fill_opacity
        if border_color is not None:
            kwargs['border_color'] = border_color
        image = vis_bbox(image,
                         box,
                         fill_color=color,
                         border_color=color,
                         thickness=thickness,
                         **kwargs)
    return image

def overlay_amodal_boxes_prediction(image,
                       annotations, oy=None, ox=None,
                       colors=None,
                       border_color=None,
                       fill_opacity=None,
                       thickness=1):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
    """
    if oy is None and ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)
    boxes = [[int(round(y)) for y in x['bbox']] for x in annotations]
    boxes = [[box[0]+ox, box[1]+oy, box[2], box[3]] for box in boxes]
    # track_ids = [x['track_id'] for x in annotations]

    sorted_inds = sorted(range(len(boxes)),
                         key=lambda i: boxes[i][2] * boxes[i][3],
                         reverse=True)

    if colors is None:
        colors = get_annotation_colors(annotations)

    for i in sorted_inds:
        box = boxes[i]
        color = colors[i]
        # track_id = track_ids[i]
        # if track_id != 9:
        #     continue
        kwargs = {}
        if fill_opacity:
            kwargs['fill_opacity'] = fill_opacity
        if border_color is not None:
            kwargs['border_color'] = border_color
        image = vis_bbox(image,
                         box,
                         fill_color=color,
                         border_color=color,
                         thickness=thickness,
                         **kwargs)
    return image

def overlay_boxes_coco(image,
                       annotations,
                       colors=None,
                       border_color=None,
                       fill_opacity=None,
                       thickness=1):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
    """
    boxes = [[int(round(y)) for y in x['bbox']] for x in annotations]
    # track_ids = [x['track_id'] for x in annotations]

    sorted_inds = sorted(range(len(boxes)),
                         key=lambda i: boxes[i][2] * boxes[i][3],
                         reverse=True)

    if colors is None:
        colors = get_annotation_colors(annotations)

    for i in sorted_inds:
        box = boxes[i]
        color = colors[i]
        # track_id = track_ids[i]
        # if track_id != 9:
        #     continue
        kwargs = {}
        if fill_opacity:
            kwargs['fill_opacity'] = fill_opacity
        if border_color is not None:
            kwargs['border_color'] = border_color
        image = vis_bbox(image,
                         box,
                         fill_color=color,
                         thickness=thickness,
                         **kwargs)
    return image


def vis_mask(image,
             mask,
             color,
             alpha=0.4,
             show_border=True,
             border_alpha=0.5,
             border_thick=1,
             border_color=None):
    """Visualizes a single binary mask."""
    image = image.astype(np.float32)
    H, W = image.shape[:2]
    mask = mask[0, :, :, None]
    idx = np.nonzero(mask)
    
    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += [alpha * x for x in color]

    if border_alpha == 0:
        return

    if border_color is None:
        border_color = [x * 0.5 for x in color]
    if isinstance(border_color, np.ndarray):
        border_color = border_color.tolist()
    contours, _ = cv2_util.findContours(mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    if border_alpha < 1:
        with_border = image.copy()
        cv2.drawContours(with_border, contours, -1, border_color, border_thick,
                         cv2.LINE_AA)
        image = ((1 - border_alpha) * image + border_alpha * with_border)
    else:
        cv2.drawContours(image, contours, -1, border_color, border_thick,
                         cv2.LINE_AA)
    return image.astype(np.uint8)


def overlay_mask_coco(image,
                      annotations,
                      alpha=0.3,
                      border_alpha=1.0,
                      border_thick=2):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        annotations (List[dict]): List of COCO annotations.
    """
    colors = colormap(rgb=True)[:len(annotations)]

    for annotation, color in zip(annotations, colors):
        mask = mask_util.decode(annotation['segmentation'])
        image = vis_mask(image,
                         mask,
                         color,
                         alpha=alpha,
                         border_alpha=border_alpha,
                         border_thick=border_thick)
    return image

def transparent_except_bbox(image, all_annos, opacity=0.6, modal=True, oy=None, ox=None):
    with_fill = np.ones_like(image) * 255
    if oy is None or ox is None:
        oy, ox = image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)

    for ann in all_annos:
        if modal and 'bbox' not in ann:
            continue
        box = ann['bbox'] if modal else ann['amodal_bbox']
        box = [max(0, box[0]+ox), max(0, box[1]+oy), min(image.shape[1], box[2]+box[0]+ox), min(image.shape[0], box[3]+box[1]+oy)]
        with_fill[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    
    image = cv2.addWeighted(with_fill, opacity, image,
                                1 - opacity, 0, image)
    return image