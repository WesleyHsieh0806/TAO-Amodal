from enum import Enum, unique


@unique
class SHAPE(Enum):
    POINT = 1
    RECT = 2
    CIRCLE = 3
    ELLIPSE = 4
    LINE = 5
    POLYLINE = 6
    POLYGON = 7
    EXTREME_BOX = 8


# in degree (used to approximate shapes using polygon)
VIA_POLYGON_SEGMENT_SUBTENDED_ANGLE = 5
