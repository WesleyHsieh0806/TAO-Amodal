def interpolate_annotations(frames, annotations, modal=True):
    """Interpolate annotations for unlabeled frames.

    Args:
        frames (List[str]): Ordered list of frames.
        annotations (Dict[str, List]): Map frame to COCO-style annotations.

    Returns:
        interpolated (Dict[str, List]): Interpolated annotations
    """
    neighbors = {}
    annotated = [i for i, f in enumerate(frames) if f in annotations]
    previous = 0
    for i in range(len(frames)):
        if previous == len(annotated) - 1:
            break
        prev_frame, next_frame = (annotated[previous], annotated[previous + 1])
        if i < prev_frame:
            curr_neighbors = (prev_frame, prev_frame)
        elif i > next_frame:
            curr_neighbors = (next_frame, next_frame)
        else:
            curr_neighbors = (prev_frame, next_frame)
        if i not in curr_neighbors:
            neighbors[i] = curr_neighbors
        else:
            if i == next_frame:
                previous += 1
    interpolated = {}
    for i, (prev_frame, next_frame) in neighbors.items():
        annotation_start = annotations[frames[prev_frame]]
        annotation_end = annotations[frames[next_frame]]
        track_start = {x['track_id']: x for x in annotation_start}
        track_end = {x['track_id']: x for x in annotation_end}
        distance = (next_frame - prev_frame)
        alpha = (next_frame - i) / distance
        anns = []
        for track in track_start:
            if track not in track_end:
                continue
            start, end = track_start[track], track_end[track]
            assert start['category_id'] == end['category_id']
            ann = {
                k: v
                for k, v in start.items()
                if k not in ('bbox', 'segmentation', 'area', 'id', 'image_id')
            }
            if modal:
                ann['bbox'] = [
                    alpha * a + (1 - alpha) * b
                    for a, b in zip(start['bbox'], end['bbox'])
                ]
            else:
                ann['amodal_bbox'] = [
                    alpha * a + (1 - alpha) * b
                    for a, b in zip(start['amodal_bbox'], end['amodal_bbox'])
                ]
            anns.append(ann)
        interpolated[frames[i]] = anns
    return interpolated
