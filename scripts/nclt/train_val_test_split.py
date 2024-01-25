from pathlib import Path

import pandas as pd
from pandas import DataFrame
import hydra
from omegaconf import DictConfig

from data_tools.split import check_in_test_set, check_in_buffer_set


@hydra.main(config_path="../../configs", config_name="split_nclt")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    track_files = sorted(list(data_dir.glob("*/track.csv")))
    column_names = ["track", "image", "pointcloud", "x", "y", "z", "roll", "pitch", "yaw"]
    train_df = DataFrame(columns=column_names)
    val_df = DataFrame(columns=column_names)
    test_df = DataFrame(columns=column_names)
    for track_file in track_files:
        track_name = track_file.parent.name
        track_df = pd.read_csv(track_file)
        track_df["track"] = track_name
        train_rows = []
        val_rows = []
        test_rows = []
        for _, row in track_df.iterrows():
            if check_in_test_set(row["x"], row["y"], test_boundary_points=cfg.p, boundary_width=cfg.p_width):
                test_rows.append(row)
                val_rows.append(row)
            elif not check_in_buffer_set(
                row["x"],
                row["y"],
                test_boundary_points=cfg.p,
                boundary_width=cfg.p_width,
                buffer_width=cfg.buffer_width,
            ):
                train_rows.append(row)
        # the difference between val and test is that test are sampled every cfg.test_step_size frames
        test_rows = test_rows[:: cfg.test_step_size]

        track_train_df = pd.DataFrame(train_rows)
        track_val_df = pd.DataFrame(val_rows)
        track_test_df = pd.DataFrame(test_rows)

        train_df = pd.concat([train_df, track_train_df], ignore_index=True)
        val_df = pd.concat([val_df, track_val_df], ignore_index=True)
        test_df = pd.concat([test_df, track_test_df], ignore_index=True)

    train_df[["image", "pointcloud"]] = train_df[["image", "pointcloud"]].astype("int64")
    val_df[["image", "pointcloud"]] = val_df[["image", "pointcloud"]].astype("int64")
    test_df[["image", "pointcloud"]] = test_df[["image", "pointcloud"]].astype("int64")

    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
