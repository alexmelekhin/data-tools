"""Methods to filter data."""

import numpy as np


def argfilter_trajectory(trajectory: np.ndarray, threshold: float) -> np.ndarray:
    """Filter a trajectory so that the minimum distance between points is above threshold.

    Args:
        trajectory: The trajectory to filter.
        threshold: The minimum distance between two points.

    Returns:
        np.ndarray: An indices of the points to keep.
    """
    indices_to_keep = [0]  # Always keep the first point

    for i in range(1, len(trajectory)):
        last_point = trajectory[indices_to_keep[-1]]
        current_point = trajectory[i]

        distance = np.linalg.norm(current_point - last_point)

        if distance >= threshold:
            indices_to_keep.append(i)

    return np.array(indices_to_keep)


def argfind_closest_timestamps(timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Find the closest timestamps in a list of timestamps.

    Args:
        timestamps: The timestamps to search through.
        target_timestamps: The timestamps to find the closest timestamps to.

    Returns:
        np.ndarray: An array of indices of the closest timestamps.
    """
    indices = []

    for target_timestamp in target_timestamps:
        closest_index = np.argmin(np.abs(timestamps - target_timestamp))
        indices.append(closest_index)

    return np.array(indices)
