"""Pointcloud processing for the NCLT dataset.

Source: https://robots.engin.umich.edu/nclt/
"""
import struct
from os import PathLike

import numpy as np


def read_velodyne_scan(scan_filepath: str | PathLike) -> np.ndarray:
    """Read a Velodyne scan from a raw binary file.

    Args:
        scan_filepath: The path to the binary file.
    """
    with open(scan_filepath, "rb") as f_bin:
        hits = []
        while True:
            x_str = f_bin.read(2)
            if x_str == b"":  # eof
                break
            x = struct.unpack("<H", x_str)[0]
            y = struct.unpack("<H", f_bin.read(2))[0]
            z = struct.unpack("<H", f_bin.read(2))[0]
            i = struct.unpack("B", f_bin.read(1))[0]  # intensity value (in range [0, 255])
            _ = struct.unpack("B", f_bin.read(1))[0]  # laser id, we don't need it
            hits.append([x, y, z, i])
    hits = np.array(hits, dtype=np.float32)

    """
    from paper:
    Due to file size considerations, we encode this data in a binary format—all values are
    in little-endian. For each point, we scale each of x, y, z, to an integer between 0 and 40 000
    by adding 100 m and discretize the result at 5 mm. For example, −90 m gets scaled to 2000.
    """
    scaling = 0.005  # 5 mm
    offset = -100.0
    hits[:, :3] = hits[:, :3] * scaling + offset
    hits[:, 3] = hits[:, 3] / 255.0  # normalize intensity to [0, 1]

    return hits
