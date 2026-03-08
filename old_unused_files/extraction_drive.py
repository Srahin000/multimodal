"""
extraction_drive.py
────────────────────────────────────────────────────────────────
One-time extraction of channel + sensor ZIPs from Google Drive
to a persistent flat-file structure (Dataset_flat/), followed by
a per-session sync to local Colab SSD (/content/dataset_cache/).

Usage on Colab (standalone):
    !python /content/drive/MyDrive/extraction_drive.py

Or import from a notebook cell:
    import sys; sys.path.insert(0, "/content/drive/MyDrive/")
    from extraction_drive import extract_all_to_drive_flat, sync_flat_to_local

── Integration context for future agents ────────────────────────────────────

NOTEBOOK: scaled_beam_only_extraction.ipynb, cell 16 (id=d733e146)
  Imports: extract_all_to_drive_flat, sync_flat_to_local, DRIVE_ROOT, DRIVE_FLAT, LOCAL_FLAT
  Passes WEATHER_CONDITIONS and TOWNS from notebook scope (don't rely on module defaults
  for those two — they may differ across experiments).

HOW ChannelDataset reads flat files (cell 11, id=412dea29):
  ChannelSampleRef has flat_path: Optional[Path] (alongside zip_path/inner_npz).
  _build_index() does: flat_dir = zp.with_suffix("")   # Town03.zip → Town03/
  Then rglobs for *_paths.npz inside that directory.
  CRITICAL: the directory we extract INTO must be named exactly {town} (e.g. Town03/)
  so that zp.with_suffix("") finds it. The ZipFile.extractall(dst_dir) call puts
  contents under dst_dir, creating dst_dir/Town03/Town03_Tjunction/cav_1/*.npz.
  _load_npz() calls np.load(ref.flat_path) directly — no ZIP overhead.

HOW SensorIndex reads flat files (cell 12, id=b1459ef3):
  SensorZipEntry has flat_dir: Optional[Path].
  _build() scans sensor_dir for subdirectories after ZIP scanning; a directory
  named "Town03_Tjunction_wiz_slope_seed42" is treated as a flat-extracted archive.
  load_sensor_frame() reads: (entry.flat_dir / cav_id / f"{frame_id}.pcd").read_bytes()
  CRITICAL: extract sensor ZIP into dst_dir / src_zip.stem (directory = ZIP stem name).

EXPECTED FLAT LAYOUT after extraction:
  Dataset_flat/
  ├── sunny/Channel Data/Nt_1_16_Nr_1_16_fc_28GHz/
  │   └── Town03/Town03/Town03_Tjunction/cav_1/*.npz   ← double Town03 from ZIP internals
  ├── sunny/Channel Data/V2I/Nt_1_16_Nr_1_16_fc_28GHz/
  │   └── ...                                           ← separate variant, same structure
  └── sunny/Sensor Data/Town03/
      └── Town03_Tjunction_wiz_slope_seed42/            ← stem name = directory name
          └── Town03_Tjunction_wiz_slope_seed42/cav_1/*.pcd + *.yaml

IDEMPOTENCY contract (do not break these checks or re-extractions will happen every run):
  Channel skip: dst_dir.exists() and any(dst_dir.rglob("*_paths.npz"))
  Sensor skip:  dst_sub.exists() and any(dst_sub.glob("**/*.pcd"))
  sync_flat_to_local skip: dst.exists() and dst.stat().st_size == src.stat().st_size
"""
import shutil
import time
import zipfile
from pathlib import Path

DRIVE_ROOT  = "/content/drive/MyDrive/Dataset"
DRIVE_FLAT  = "/content/drive/MyDrive/Dataset_flat"
LOCAL_FLAT  = "/content/dataset_cache"
CONFIG_NAME = "Nt_1_16_Nr_1_16_fc_28GHz"
WEATHER_CONDITIONS = ["sunny", "foggy", "rainy"]
TOWNS = ["Town03", "Town05", "Town07", "Town10"]


def extract_all_to_drive_flat(
    drive_root=DRIVE_ROOT,
    flat_root=DRIVE_FLAT,
    weather_conditions=WEATHER_CONDITIONS,
    towns=TOWNS,
    config_name=CONFIG_NAME,
):
    """
    ONE-TIME: Extract channel + sensor ZIPs from Dataset/ → Dataset_flat/ in Drive.
    Idempotent — skips towns already fully extracted. Safe to re-run after crashes.

    Channel ZIPs:  Dataset/{weather}/Channel Data/{config}/Town03.zip
      → Dataset_flat/{weather}/Channel Data/{config}/Town03/Town03/...npz

    Sensor ZIPs:   Dataset/{weather}/Sensor Data/{town}/<scenario>.zip
      → Dataset_flat/{weather}/Sensor Data/{town}/<scenario>/<scenario>/...pcd/.yaml
    """
    CHANNEL_VARIANTS = ["Channel Data", "Channel Data/V2I"]
    SENSOR_VARIANTS  = ["Sensor Data"]
    dr, fr = Path(drive_root), Path(flat_root)
    _t = time.time()

    for weather in weather_conditions:
        # ── Channel ZIPs ──────────────────────────────────────────────────────
        for variant in CHANNEL_VARIANTS:
            src_cfg = dr / weather / variant / config_name
            dst_cfg = fr / weather / variant / config_name
            if not src_cfg.exists():
                continue
            for town in towns:
                src_zip = src_cfg / f"{town}.zip"
                dst_dir = dst_cfg / town
                if not src_zip.exists():
                    continue
                if dst_dir.exists() and any(dst_dir.rglob("*_paths.npz")):
                    print(f"  [skip] channel {weather}/{town}")
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Extracting channel {weather}/{town}.zip "
                      f"({src_zip.stat().st_size / 1e6:.0f} MB) …")
                with zipfile.ZipFile(src_zip) as zf:
                    zf.extractall(dst_dir)

        # ── Sensor ZIPs ───────────────────────────────────────────────────────
        for variant in SENSOR_VARIANTS:
            for town in towns:
                src_dir = dr / weather / variant / town
                dst_dir = fr / weather / variant / town
                if not src_dir.exists():
                    continue
                for src_zip in sorted(src_dir.glob("*.zip")):
                    dst_sub = dst_dir / src_zip.stem
                    if dst_sub.exists() and any(dst_sub.glob("**/*.pcd")):
                        print(f"  [skip] sensor {weather}/{town}/{src_zip.stem}")
                        continue
                    dst_sub.mkdir(parents=True, exist_ok=True)
                    print(f"  Extracting sensor {weather}/{town}/{src_zip.name} "
                          f"({src_zip.stat().st_size / 1e6:.0f} MB) …")
                    with zipfile.ZipFile(src_zip) as zf:
                        zf.extractall(dst_sub)

    print(f"Extraction complete in {time.time() - _t:.0f}s.")


def sync_flat_to_local(flat_root=DRIVE_FLAT, local_root=LOCAL_FLAT):
    """
    PER-SESSION: Copy flat files from Drive (flat_root) → local Colab SSD (local_root).
    Skips files already present with matching file size — re-runs are near-instant.
    Typically 1–5 min for a full dataset vs re-extracting ZIPs every crash.
    """
    _t, copied, skipped = time.time(), 0, 0
    for src in sorted(Path(flat_root).rglob("*")):
        if src.is_dir():
            continue
        dst = Path(local_root) / src.relative_to(flat_root)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Sync done in {time.time() - _t:.1f}s — "
          f"{copied} copied, {skipped} already cached")


if __name__ == "__main__":
    print("=== Drive flat extraction ===")
    extract_all_to_drive_flat()
    print("\n=== Sync to local SSD ===")
    sync_flat_to_local()
    print(f"\nData ready at: {LOCAL_FLAT}")
