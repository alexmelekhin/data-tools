import concurrent.futures
from os import PathLike
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from data_tools.filtering import argfilter_trajectory, argfind_closest_timestamps
from data_tools.nclt.image import Undistort, center_crop
from data_tools.nclt.pointcloud import read_velodyne_scan

matplotlib.use("Agg")


class Preprocessor:
    def __init__(
        self,
        data_dir: PathLike,
        output_dir: PathLike,
        tracklist: list[str],
        distance_threshold: float = 1.0,
        cameras: list[str] = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5"],
        num_threads: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.tracklist = tracklist
        self.distance_threshold = distance_threshold
        self.cameras = cameras
        self.num_threads = num_threads

        self.images_dirname = "images"
        self.lidar_dirname = "velodyne_data"

        self.cameras_synced = True
        self.lidar_synced_with_cameras = True
        self.img_format = "tiff"

        self.undistorter = {}
        for cam in self.cameras:
            undistort_map_filepath = self.data_dir / "undistort_maps" / f"U2D_{cam}_1616X1232.txt"
            self.undistorter[cam] = Undistort(undistort_map_filepath)

        self.center_crop_size = (768, 960)
        self.resize_size = (384, 480)

    def get_images_input_dir(self, track: str) -> Path:
        return self.data_dir / "images" / track / "lb3"

    def get_lidar_input_dir(self, track: str) -> Path:
        return self.data_dir / "velodyne_data" / track / "velodyne_sync"

    def read_track_df(self, track: str) -> pd.DataFrame:
        track_df = pd.read_csv(
            self.data_dir / "ground_truth" / f"groundtruth_{track}.csv",
            header=None,
            low_memory=False,
            skiprows=20,  # first frames are useless
        )
        track_df.columns = ["timestamp", "x", "y", "z", "roll", "pitch", "yaw"]
        track_df[["x", "y", "z", "roll", "pitch", "yaw"]] = track_df[
            ["x", "y", "z", "roll", "pitch", "yaw"]
        ].astype(np.float64)
        return track_df

    def process_image(self, src_filepath: Path, dst_filepath: Path, cam: str) -> None:
        dst_filepath.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src_filepath))
        img = self.undistorter[cam](img)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = center_crop(img, self.center_crop_size)
        img = cv2.resize(img, self.resize_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(dst_filepath), img)

    def process_lidar(self, src_filepath: Path, dst_filepath: Path) -> None:
        dst_filepath.parent.mkdir(parents=True, exist_ok=True)
        pc = read_velodyne_scan(src_filepath)
        pc.tofile(dst_filepath)

    def plot_track_map(self, utms: np.ndarray) -> np.ndarray:
        x, y = utms[:, 0], utms[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(y, x, s=0.5, c="blue")
        ax.set_xlabel("y")
        ax.set_xlim(-750, 150)
        ax.set_ylabel("x")
        ax.set_ylim(-380, 140)
        ax.set_aspect("equal", adjustable="box")
        fig.canvas.draw()
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height(physical=True)[::-1] + (3,))
        plt.close(fig)
        # convert from RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def run(self) -> None:
        for track in self.tracklist:
            self.process_track(track)

    def process_track(self, track: str) -> None:
        track_dir = self.output_dir / track
        if not track_dir.exists():
            track_dir.mkdir(parents=True)

        track_df = self.read_track_df(track)
        i_filtered_trajectory = argfilter_trajectory(
            track_df[["x", "y", "z"]].values, threshold=self.distance_threshold
        )
        track_df = track_df.iloc[i_filtered_trajectory].reset_index(drop=True)

        images_input_dir = self.get_images_input_dir(track)
        images_output_dir = track_dir / self.images_dirname
        if not images_output_dir.exists():
            images_output_dir.mkdir(parents=True)

        if self.cameras_synced:
            images_list = sorted(
                [
                    x
                    for x in (images_input_dir / self.cameras[0]).iterdir()
                    if x.suffix == f".{self.img_format}"
                ]
            )
            images_timestamps = np.array([int(x.stem) for x in images_list])
            i_filtered_images = argfind_closest_timestamps(images_timestamps, track_df["timestamp"].values)
            images_timestamps = images_timestamps[i_filtered_images]
            logger.info(
                f"Selected {len(images_timestamps)} images for track {track} (out of {len(images_list)}))"
            )
            images_ts_diffs = np.abs(images_timestamps - track_df["timestamp"].values)
            logger.info(
                f"ts_diff (min / mean / max): {(images_ts_diffs.min() / 1000):.2f} "
                f"/ {(images_ts_diffs.mean() / 1000):.2f} / {(images_ts_diffs.max() / 1000):.2f} ms"
            )
            track_df["image"] = images_timestamps
        else:
            raise NotImplementedError()

        lidar_input_dir = self.get_lidar_input_dir(track)
        lidar_output_dir = track_dir / self.lidar_dirname
        if not lidar_output_dir.exists():
            lidar_output_dir.mkdir(parents=True)

        if self.lidar_synced_with_cameras:
            lidar_timestamps = images_timestamps
            # check if all files exists
            missing_lidar_indices = []
            for i, lidar_ts in enumerate(lidar_timestamps):
                if not (lidar_input_dir / f"{lidar_ts}.bin").exists():
                    missing_lidar_indices.append(i)
        else:
            raise NotImplementedError()
        track_df["pointcloud"] = lidar_timestamps

        logger.warning(f"Missing {len(missing_lidar_indices)} lidar scans for track {track}. Dropping them.")
        track_df.drop(missing_lidar_indices, inplace=True)
        images_timestamps = track_df["image"].to_numpy()
        lidar_timestamps = track_df["pointcloud"].to_numpy()

        track_df = track_df[["timestamp", "image", "pointcloud", "x", "y", "z", "roll", "pitch", "yaw"]]
        track_df.to_csv(track_dir / "track.csv", index=False)

        track_map = self.plot_track_map(track_df[["x", "y"]].to_numpy())
        cv2.imwrite(str(track_dir / "track_map.png"), track_map)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            logger.info(f"Processing images for track {track}")
            for camera in tqdm(self.cameras, desc="Camera", leave=False, position=0):
                futures = []
                for image_ts in images_timestamps:
                    image_path = images_input_dir / camera / f"{image_ts}.tiff"
                    image_path_output = images_output_dir / camera / f"{image_ts}.png"
                    futures.append(executor.submit(self.process_image, image_path, image_path_output, camera))
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    desc=camera,
                    total=len(futures),
                    position=1,
                    leave=False,
                ):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(e)

            logger.info(f"Processing lidar scans for track {track}")
            futures = []
            for lidar_ts in lidar_timestamps:
                lidar_path = lidar_input_dir / f"{lidar_ts}.bin"
                lidar_path_output = lidar_output_dir / f"{lidar_ts}.bin"
                futures.append(executor.submit(self.process_lidar, lidar_path, lidar_path_output))
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Lidar",
                total=len(futures),
                position=0,
                leave=False,
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.error(e)

        logger.info(f"Track {track} processed successfully")
