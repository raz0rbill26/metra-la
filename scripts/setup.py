#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def download_and_setup(dataset_name: str = "annnnguyen/metr-la-dataset",
                       target_dir: Path = DEFAULT_DATA_DIR,
                       use_symlink: bool = False,
                       force: bool = False) -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    expected_files = ["METR-LA.h5", "adj_METR-LA.pkl"]
    already_setup = all((target_path / f).exists() for f in expected_files)

    if already_setup and not force:
        print(f"Dataset already present in {target_path}")
        return target_path

    print(f"Downloading {dataset_name} via kagglehub...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(dataset_name)
    except Exception as e:
        print(f"Failed to download dataset: {e}", file=sys.stderr)
        sys.exit(1)

    src_dir = Path(downloaded_path)
    for filename in expected_files:
        matches = list(src_dir.rglob(filename))
        if not matches:
            print(f"Missing file {filename} in {src_dir}", file=sys.stderr)
            sys.exit(1)

        src_file = matches[0]
        dst_file = target_path / filename
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()

        if use_symlink:
            try:
                dst_file.symlink_to(src_file)
            except OSError:
                shutil.copy2(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)

    print(f"Dataset setup complete at {target_path}")
    return target_path


def verify_dataset(target_dir: Path):
    h5_path = target_dir / "METR-LA.h5"
    pkl_path = target_dir / "adj_METR-LA.pkl"

    if not h5_path.exists() or not pkl_path.exists():
        print(f"Verification failed: files missing in {target_dir}", file=sys.stderr)
        return False

    import h5py
    import pickle

    with h5py.File(h5_path, "r") as f:
        shape = f["df/block0_values"].shape
        print(f"HDF5 shape: {shape}")

    with open(pkl_path, "rb") as f:
        try:
            adj = pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            adj = pickle.load(f)
        print(f"Adjacency matrix shape: {adj[2].shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download and setup METR-LA dataset")
    parser.add_argument("--dataset", type=str, default="annnnguyen/metr-la-dataset")
    parser.add_argument("--target-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_dir = download_and_setup(
        dataset_name=args.dataset,
        target_dir=Path(args.target_dir),
        use_symlink=args.symlink,
        force=args.force
    )
    verify_dataset(target_dir)


if __name__ == "__main__":
    main()
