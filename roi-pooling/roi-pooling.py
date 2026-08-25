import math

def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    S = output_size
    result = []

    for x1, y1, x2, y2 in rois:
        H_R = y2 - y1          # region height  (rows y1 .. y2-1)
        W_R = x2 - x1          # region width   (cols x1 .. x2-1)

        grid = []
        for i in range(S):
            # vertical (row) bounds for output row i
            h_start = y1 + (i * H_R) // S
            h_end   = y1 + ((i + 1) * H_R) // S
            if h_end <= h_start:        # empty bin -> expand to one row
                h_end = h_start + 1

            out_row = []
            for j in range(S):
                # horizontal (col) bounds for output col j
                w_start = x1 + (j * W_R) // S
                w_end   = x1 + ((j + 1) * W_R) // S
                if w_end <= w_start:    # empty bin -> expand to one col
                    w_end = w_start + 1

                # max over the (now non-empty) bin
                best = None
                for r in range(h_start, h_end):
                    for c in range(w_start, w_end):
                        val = feature_map[r][c]
                        if best is None or val > best:
                            best = val
                out_row.append(best)
            grid.append(out_row)

        result.append(grid)

    return result
