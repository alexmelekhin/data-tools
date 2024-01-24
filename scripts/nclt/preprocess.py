from pathlib import Path

import hydra
from omegaconf import DictConfig

from data_tools.nclt.preprocessor import Preprocessor

TRACKLIST = [
    "2012-01-08",
]


@hydra.main(config_path="../../configs/", config_name="preprocess_nclt")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    tracklist = cfg.tracklist
    distance_threshold = cfg.distance_threshold
    cameras = cfg.cameras
    num_threads = cfg.num_threads

    preprocessor = Preprocessor(
        data_dir,
        output_dir,
        tracklist,
        distance_threshold,
        cameras,
        num_threads,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
