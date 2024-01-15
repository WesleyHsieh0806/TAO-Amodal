import cv2
import numpy as np


def create_color_map():
    # This function has been copied with minor changes from the DAVIS dataset API at:
    # https://github.com/davisvideochallenge/davis2017-evaluation (Caelles, Pont-Tuset, et al.)
    N = 256

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap.tolist()


def overlay_mask_on_image(image, mask, opacity, color):
    assert mask.ndim == 2
    mask_bgr = np.stack((mask, mask, mask), axis=2)
    masked_image = np.where(mask_bgr > 0, color, image)
    return ((opacity * masked_image) + ((1. - opacity) * image)).astype(np.uint8)


def bbox_from_mask(mask):
    reduced_y = np.any(mask, axis=0)
    reduced_x = np.any(mask, axis=1)

    x_min = reduced_y.argmax()
    if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
        return None

    x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

    y_min = reduced_x.argmax()
    y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

    return x_min, y_min, x_max, y_max


def annotate_image(image, mask, color, label, point=None, **kwargs):
    """
    :param image: np.ndarray(H, W, 3)
    :param mask: np.ndarray(H, W)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param label: str
    :param kwargs: "bbox_thickness", "text_font", "font_size", "mask_opacity"
    :return: np.ndarray(H, W, 3)
    """
    annotated_image = overlay_mask_on_image(image, mask, color=color, opacity=kwargs.get("mask_opacity", 0.5))
    xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(mask)]

    bbox_thickness = kwargs.get("bbox_thickness", 2)
    text_font = kwargs.get("text_font", cv2.FONT_HERSHEY_SIMPLEX)
    font_size = kwargs.get("font_size", 0.5)

    annotated_image = cv2.rectangle(cv2.UMat(annotated_image), (xmin, ymin), (xmax, ymax), color=tuple(color),
                                    thickness=bbox_thickness)

    (text_width, text_height), _ = cv2.getTextSize(label, text_font, font_size, thickness=1)
    text_offset_x, text_offset_y = int(xmin + 2), int(ymin + text_height + 2)

    text_bg_box_pt1 = int(text_offset_x), int(text_offset_y + 2)
    text_bg_box_pt2 = int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)

    annotated_image = cv2.rectangle(cv2.UMat(annotated_image), text_bg_box_pt1, text_bg_box_pt2, color=(255, 255, 255), thickness=-1)
    annotated_image = cv2.putText(cv2.UMat(annotated_image), label, (text_offset_x, text_offset_y), text_font, font_size, (0, 0, 0))

    if point is not None:
        # use a darker color so the point is more visible on the mask
        color = tuple([int(round(0.5 * c)) for c in color])
        annotated_image = cv2.circle(cv2.UMat(annotated_image), point, radius=3, color=color, thickness=-1)

    if isinstance(annotated_image, cv2.UMat):
        # sometimes OpenCV functions return objects of type cv2.UMat instead of numpy arrays
        return annotated_image.get()
    else:
        return annotated_image

