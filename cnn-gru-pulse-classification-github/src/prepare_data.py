"""Prepare pulse windows and metadata without training a model.

Run from the project root: python -m src.prepare_data
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocessing import extract_windows, read_pulse_file


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_dataset(config_path):
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    raw_dir = Path(config["raw_data_dir"])
    output_dir = Path(config["dataset_output_dir"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {raw_dir}")

    points = int(config.get("sequence_length", 500))
    peak_width = int(config.get("peak_width", 20))
    class_dirs = sorted(
        [path for path in raw_dir.iterdir()
         if path.is_dir() and not path.name.startswith(".")],
        key=lambda path: path.name,
    )
    expected_classes = int(config["num_classes"])
    if len(class_dirs) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} class folders, found {len(class_dirs)}."
        )

    class_to_index = {path.name: index for index, path in enumerate(class_dirs)}
    feature_batches, label_batches = [], []
    record_id_batches, window_index_batches = [], []
    records, class_counts = [], []

    for class_dir in class_dirs:
        label = class_to_index[class_dir.name]
        files = sorted(
            [path for path in class_dir.iterdir()
             if path.is_file() and path.suffix.lower() == ".txt"],
            key=lambda path: path.name,
        )
        if not files:
            raise ValueError(f"No TXT recordings found in {class_dir}")

        sample_count = 0
        for file_path in files:
            record_id = len(records)
            signals = read_pulse_file(file_path)
            windows = extract_windows(signals, points=points, peak_width=peak_width)
            count = len(windows)
            records.append({
                "record_id": record_id,
                "class_id": label,
                "class_name": class_dir.name,
                "source": file_path.relative_to(raw_dir).as_posix(),
                "valid_frames": len(signals),
                "num_windows": count,
            })
            print(f"{class_dir.name}: record {record_id}, {count} windows")

            if count:
                feature_batches.append(windows)
                label_batches.append(np.full(count, label, dtype=np.int64))
                record_id_batches.append(np.full(count, record_id, dtype=np.int64))
                window_index_batches.append(np.arange(count, dtype=np.int64))
                sample_count += count

        if not sample_count:
            raise ValueError(
                f"No waveform windows extracted for class {class_dir.name}. "
                "Check recording lengths and peak detection settings."
            )
        class_counts.append({
            "class_id": label,
            "class_name": class_dir.name,
            "num_records": len(files),
            "num_windows": sample_count,
        })

    features = np.concatenate(feature_batches).astype(np.float32, copy=False)
    labels = np.concatenate(label_batches)
    record_ids = np.concatenate(record_id_batches)
    window_indices = np.concatenate(window_index_batches)
    summary = {
        "num_classes": len(class_dirs),
        "num_records": len(records),
        "num_samples": len(labels),
        "feature_shape": list(features.shape),
        "classes": class_counts,
        "single_record_classes": [item["class_name"] for item in class_counts
                                  if item["num_records"] == 1],
        "split_note": (
            "No train/test split has been created. Windows overlap. "
            "Use recording/subject metadata or a documented temporal split "
            "with overlap removal. Single-record classes limit independent "
            "recording-level evaluation."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pulse_windows.npz",
        features=features,
        labels=labels,
        record_ids=record_ids,
        window_indices=window_indices,
    )
    write_json(output_dir / "class_to_index.json", class_to_index)
    write_json(output_dir / "records.json", records)
    write_json(output_dir / "summary.json", summary)

    print(f"\nClasses: {len(class_dirs)}")
    print(f"Recordings: {len(records)}")
    print(f"Feature shape: {features.shape}")
    print(f"Saved to: {output_dir}")
    if summary["single_record_classes"]:
        print("Note: some classes have only one recording; review the evaluation split.")
    return summary


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=project_root / "config.local.json",
        help="Local JSON configuration file.",
    )
    args = parser.parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()
