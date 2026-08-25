def generate_anchors(feature_size: int, image_size: float,
                     scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Generate anchor boxes for object detection.
    """
    import math                                  # for sqrt()

    stride = image_size / feature_size           # STEP 1: pixels per grid cell
    anchors = []                                 # result list

    # STEP 2 loop: walk the grid in row-major order (i = row, j = column)
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride              # center x of this cell
            cy = (i + 0.5) * stride              # center y of this cell

            # STEP 3 loop: every scale, and inside it every aspect ratio
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)         # box width
                    h = s / math.sqrt(r)         # box height

                    # STEP 4: center+size -> [x1, y1, x2, y2]
                    anchors.append([
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                    ])

    return anchors
