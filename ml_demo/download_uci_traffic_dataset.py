from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/492/metro%2Binterstate%2Btraffic%2Bvolume.zip"
ZIP_MEMBER_NAME = "Metro_Interstate_Traffic_Volume.csv.gz"
OUTPUT_FILENAME = "metro_interstate_traffic_volume.csv.gz"


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / "uci_traffic_dataset.zip"

        print(f"Downloading dataset archive from {DATASET_ZIP_URL}")
        urllib.request.urlretrieve(DATASET_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(ZIP_MEMBER_NAME, path=temp_dir_path)

        extracted_path = temp_dir_path / ZIP_MEMBER_NAME
        shutil.copy2(extracted_path, output_path)

    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
