import math
from .constants import SHAPE, VIA_POLYGON_SEGMENT_SUBTENDED_ANGLE


def _points_to_bbox(points):
    xs = points[::2]
    ys = points[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def via_shape_to_coco(via_pts, include_segmentation=False):
    """Convert via shape to COCO annotation.
    
    Modified from via_region_shape_to_coco_annotation in via 2.x.y:
    https://gitlab.com/vgg/via/blob/a838c9b571e76f86e496d01d50fd289cb0e23726/via-2.x.y/src/via.js#L1266
    """
    annotation = {'segmentation': [], 'area': [], 'bbox': [], 'iscrowd': 0}
    try:
        shape_type = SHAPE(via_pts[0])
    except ValueError:
        raise ValueError(f'Unknown shape type: {shape_type}')
    annotation['via_shape_type'] = shape_type.name
    points = via_pts[1:]

    if shape_type == SHAPE.RECT:
        x0, y0, w, h = points
        if include_segmentation:
            x1, y1 = x0 + w, y0 + h
            annotation['segmentation'] = [[x0, y0, x1, y0, x1, y1, x0, y1]]
        annotation['area'] = w * h
        annotation['bbox'] = [x0, y0, w, h]
    elif shape_type in {SHAPE.CIRCLE, SHAPE.ELLIPSE}:
        cx, cy, rx = points[:3]
        if shape_type == SHAPE.CIRCLE:
            ry = rx
        else:
            ry = points[4]

        theta_to_radian = math.PI / 180
        segmentation = []
        for theta in range(0, 360, VIA_POLYGON_SEGMENT_SUBTENDED_ANGLE):
            theta_radian = theta * theta_to_radian
            x = cx + rx * math.cos(theta_radian)
            y = cy + ry * math.sin(theta_radian)
            segmentation.append(x, y)
        annotation['bbox'] = _points_to_bbox(segmentation)
        if include_segmentation:
            annotation['segmentation'] = [segmentation]
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    elif shape_type in {SHAPE.POLYGON, SHAPE.EXTREME_BOX}:
        if include_segmentation:
            annotation['segmentation'] = [points.copy()]
        annotation['bbox'] = _points_to_bbox(points)
        w, h = annotation['bbox'][-2:]
        annotation['area'] = w * h  # approximate area
    else:
        raise NotImplementedError(f'Unknown shape type: {shape_type.name}')
    return annotation
