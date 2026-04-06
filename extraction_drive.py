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


def verify_flat_extraction(
    drive_root=DRIVE_ROOT,
    flat_root=DRIVE_FLAT,
    weather_conditions=WEATHER_CONDITIONS,
    towns=TOWNS,
    config_name=CONFIG_NAME,
):
    """
    Scan Dataset_flat/ and report any folders that exist but are missing expected
    content (channel: *_paths.npz; sensor: **/*.pcd).  Prints a summary table and
    returns two lists: (empty_channel, empty_sensor) with paths that need re-extraction.
    """
    CHANNEL_VARIANTS = ["Channel Data", "Channel Data/V2I"]
    dr, fr = Path(drive_root), Path(flat_root)
    empty_channel, empty_sensor = [], []

    print(f"\n{'─'*60}")
    print("Verifying flat extraction …")
    print(f"{'─'*60}")

    for weather in weather_conditions:
        # ── Channel folders ───────────────────────────────────────────────────
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
                if not dst_dir.exists():
                    status = "MISSING DIR"
                    empty_channel.append(dst_dir)
                else:
                    npz_files = list(dst_dir.rglob("*_paths.npz"))
                    if npz_files:
                        status = f"OK ({len(npz_files)} npz)"
                    else:
                        # Folder exists but no npz — check if any files at all
                        all_files = list(dst_dir.rglob("*"))
                        all_files = [f for f in all_files if f.is_file()]
                        status = f"EMPTY ({len(all_files)} other files)" if all_files else "EMPTY (no files)"
                        empty_channel.append(dst_dir)
                        # Show ZIP contents to help diagnose structure mismatch
                        try:
                            with zipfile.ZipFile(src_zip) as zf:
                                names = zf.namelist()[:5]
                            status += f" | zip top-level: {names}"
                        except Exception:
                            pass
                print(f"  channel {weather:8s} {variant:20s} {town}: {status}")

        # ── Sensor folders ────────────────────────────────────────────────────
        for town in towns:
            src_dir = dr / weather / "Sensor Data" / town
            dst_dir = fr / weather / "Sensor Data" / town
            if not src_dir.exists():
                continue
            for src_zip in sorted(src_dir.glob("*.zip")):
                dst_sub = dst_dir / src_zip.stem
                if not dst_sub.exists():
                    status = "MISSING DIR"
                    empty_sensor.append(dst_sub)
                else:
                    pcd_files = list(dst_sub.glob("**/*.pcd"))
                    if pcd_files:
                        status = f"OK ({len(pcd_files)} pcd)"
                    else:
                        all_files = [f for f in dst_sub.rglob("*") if f.is_file()]
                        status = f"EMPTY ({len(all_files)} other files)" if all_files else "EMPTY (no files)"
                        empty_sensor.append(dst_sub)
                        try:
                            with zipfile.ZipFile(src_zip) as zf:
                                names = zf.namelist()[:5]
                            status += f" | zip top-level: {names}"
                        except Exception:
                            pass
                print(f"  sensor  {weather:8s} Sensor Data/{town:8s} {src_zip.stem}: {status}")

    print(f"\n{'─'*60}")
    print(f"Verification complete — {len(empty_channel)} channel + {len(empty_sensor)} sensor folders need re-extraction")
    if empty_channel:
        print("  Empty channel folders:")
        for p in empty_channel:
            print(f"    {p}")
    if empty_sensor:
        print("  Empty sensor folders:")
        for p in empty_sensor:
            print(f"    {p}")
    print(f"{'─'*60}\n")
    return empty_channel, empty_sensor


def repair_flat_extraction(
    drive_root=DRIVE_ROOT,
    flat_root=DRIVE_FLAT,
    weather_conditions=WEATHER_CONDITIONS,
    towns=TOWNS,
    config_name=CONFIG_NAME,
    dry_run=False,
):
    """
    Check every flat folder against its source ZIP and repair any that are bad.

    A folder is considered BAD if:
      - It contains .zip files (stale Drive sync artefacts), OR
      - It is missing the expected file type
          channel → *_paths.npz
          sensor  → **/*.pcd

    Bad folders are wiped (shutil.rmtree) and then re-extracted from the
    matching source ZIP.  Good folders are left completely untouched.

    dry_run=True: report problems but do NOT delete or re-extract anything.

    Usage on Colab:
        from extraction_drive import repair_flat_extraction
        repair_flat_extraction()          # live repair
        repair_flat_extraction(dry_run=True)  # preview only
    """
    CHANNEL_VARIANTS = ["Channel Data", "Channel Data/V2I"]
    dr, fr = Path(drive_root), Path(flat_root)
    repaired, skipped, missing_src = 0, 0, 0

    def _is_bad_channel(folder: Path) -> tuple[bool, str]:
        """Return (is_bad, reason)."""
        stray_zips = list(folder.rglob("*.zip"))
        if stray_zips:
            return True, f"contains {len(stray_zips)} unexpected .zip file(s): {[z.name for z in stray_zips[:3]]}"
        npz_files = list(folder.rglob("*_paths.npz"))
        if not npz_files:
            return True, "no *_paths.npz files found"
        return False, f"OK ({len(npz_files)} npz)"

    def _is_bad_sensor(folder: Path) -> tuple[bool, str]:
        stray_zips = list(folder.rglob("*.zip"))
        if stray_zips:
            return True, f"contains {len(stray_zips)} unexpected .zip file(s): {[z.name for z in stray_zips[:3]]}"
        pcd_files = list(folder.glob("**/*.pcd"))
        if not pcd_files:
            return True, "no .pcd files found"
        return False, f"OK ({len(pcd_files)} pcd)"

    def _wipe_and_extract(src_zip: Path, dst_dir: Path, label: str):
        nonlocal repaired
        if dry_run:
            print(f"  [DRY RUN] would wipe + re-extract: {label}")
            return
        print(f"  [repair] wiping {dst_dir} …")
        shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [repair] extracting {src_zip.name} ({src_zip.stat().st_size / 1e6:.0f} MB) …")
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst_dir)
        npz_count = len(list(dst_dir.rglob("*_paths.npz")))
        pcd_count = len(list(dst_dir.glob("**/*.pcd")))
        print(f"    → {npz_count} npz, {pcd_count} pcd after re-extraction")
        repaired += 1

    print(f"\n{'─'*64}")
    print(f"repair_flat_extraction — {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'─'*64}")

    # ── Channel folders ───────────────────────────────────────────────────────
    for weather in weather_conditions:
        for variant in CHANNEL_VARIANTS:
            src_cfg = dr / weather / variant / config_name
            dst_cfg = fr / weather / variant / config_name
            if not src_cfg.exists():
                continue
            for town in towns:
                src_zip = src_cfg / f"{town}.zip"
                dst_dir = dst_cfg / town
                label   = f"channel {weather}/{variant}/{town}"

                if not src_zip.exists():
                    print(f"  [WARN] source zip missing — skipping: {src_zip}")
                    missing_src += 1
                    continue

                if not dst_dir.exists():
                    # Never extracted — just extract now
                    print(f"  [missing] {label} — extracting for the first time")
                    if not dry_run:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(src_zip) as zf:
                            zf.extractall(dst_dir)
                        repaired += 1
                    else:
                        print(f"  [DRY RUN] would extract: {label}")
                    continue

                bad, reason = _is_bad_channel(dst_dir)
                if bad:
                    print(f"  [BAD] {label} — {reason}")
                    _wipe_and_extract(src_zip, dst_dir, label)
                else:
                    print(f"  [ok]  {label} — {reason}")
                    skipped += 1

    # ── Sensor folders ────────────────────────────────────────────────────────
    for weather in weather_conditions:
        src_base = dr / weather / "Sensor Data"
        dst_base = fr / weather / "Sensor Data"
        if not src_base.exists():
            continue
        for town in towns:
            src_town = src_base / town
            if not src_town.exists():
                continue
            for src_zip in sorted(src_town.glob("*.zip")):
                dst_sub = dst_base / town / src_zip.stem
                label   = f"sensor  {weather}/{town}/{src_zip.stem}"

                if not dst_sub.exists():
                    print(f"  [missing] {label} — extracting for the first time")
                    if not dry_run:
                        dst_sub.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(src_zip) as zf:
                            zf.extractall(dst_sub)
                        repaired += 1
                    else:
                        print(f"  [DRY RUN] would extract: {label}")
                    continue

                bad, reason = _is_bad_sensor(dst_sub)
                if bad:
                    print(f"  [BAD] {label} — {reason}")
                    _wipe_and_extract(src_zip, dst_sub, label)
                else:
                    print(f"  [ok]  {label} — {reason}")
                    skipped += 1

    print(f"\n{'─'*64}")
    print(f"Done — {repaired} repaired, {skipped} ok, {missing_src} missing source zips")
    if dry_run:
        print("(dry run — nothing was changed)")
    print(f"{'─'*64}\n")


def extract_all_to_drive_flat(
    drive_root=DRIVE_ROOT,
    flat_root=DRIVE_FLAT,
    weather_conditions=WEATHER_CONDITIONS,
    towns=TOWNS,
    config_name=CONFIG_NAME,
    force_reextract=False,
):
    """
    ONE-TIME: Extract channel + sensor ZIPs from Dataset/ → Dataset_flat/ in Drive.
    Idempotent — skips towns already fully extracted. Safe to re-run after crashes.

    Channel ZIPs:  Dataset/{weather}/Channel Data/{config}/Town03.zip
      → Dataset_flat/{weather}/Channel Data/{config}/Town03/Town03/...npz

    Sensor ZIPs:   Dataset/{weather}/Sensor Data/{town}/<scenario>.zip
      → Dataset_flat/{weather}/Sensor Data/{town}/<scenario>/<scenario>/...pcd/.yaml

    force_reextract=True: re-extract even if dst_dir exists (use after verify reveals empty folders).
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
                # Skip only if folder exists AND has actual npz content
                if not force_reextract and dst_dir.exists() and any(dst_dir.rglob("*_paths.npz")):
                    print(f"  [skip] channel {weather}/{town}")
                    continue
                if dst_dir.exists() and not any(dst_dir.rglob("*_paths.npz")):
                    print(f"  [re-extract] channel {weather}/{town} — folder exists but no npz found")
                dst_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Extracting channel {weather}/{town}.zip "
                      f"({src_zip.stat().st_size / 1e6:.0f} MB) …")
                with zipfile.ZipFile(src_zip) as zf:
                    zf.extractall(dst_dir)
                npz_count = len(list(dst_dir.rglob("*_paths.npz")))
                print(f"    → {npz_count} npz files extracted")
                if npz_count == 0:
                    print(f"    WARNING: still no npz after extraction — check ZIP structure above")

        # ── Sensor ZIPs ───────────────────────────────────────────────────────
        for variant in SENSOR_VARIANTS:
            for town in towns:
                src_dir = dr / weather / variant / town
                dst_dir = fr / weather / variant / town
                if not src_dir.exists():
                    continue
                for src_zip in sorted(src_dir.glob("*.zip")):
                    dst_sub = dst_dir / src_zip.stem
                    if not force_reextract and dst_sub.exists() and any(dst_sub.glob("**/*.pcd")):
                        print(f"  [skip] sensor {weather}/{town}/{src_zip.stem}")
                        continue
                    if dst_sub.exists() and not any(dst_sub.glob("**/*.pcd")):
                        print(f"  [re-extract] sensor {weather}/{town}/{src_zip.stem} — no pcd found")
                    dst_sub.mkdir(parents=True, exist_ok=True)
                    print(f"  Extracting sensor {weather}/{town}/{src_zip.name} "
                          f"({src_zip.stat().st_size / 1e6:.0f} MB) …")
                    with zipfile.ZipFile(src_zip) as zf:
                        zf.extractall(dst_sub)
                    pcd_count = len(list(dst_sub.glob("**/*.pcd")))
                    print(f"    → {pcd_count} pcd files extracted")

    print(f"Extraction complete in {time.time() - _t:.0f}s.")


def sync_channel_flat_to_ssd(
    drive_root=DRIVE_ROOT,
    local_root=LOCAL_FLAT,
    weather_conditions=WEATHER_CONDITIONS,
    towns=TOWNS,
    config_name=CONFIG_NAME,
):
    """
    PER-SESSION: Copy extracted channel flat directories from Drive to Colab SSD.

    Source:  {drive_root}/{weather}/{variant}/{config}/{town}/   (folder next to Town03.zip)
    Dest:    {local_root}/{weather}/{variant}/{config}/{town}/

    Skips files already present with matching size. Channel data is ~120 MB total
    and syncs in seconds. After sync, ChannelDataset._load_npz() will auto-use
    the fast local flat files instead of Drive ZIPs.

    For sensor data (multimodal), call sync_flat_to_local() on the full Dataset_flat
    or pass sensor=True here — sensor flat dirs are much larger (~GB each).
    """
    CHANNEL_VARIANTS = ["Channel Data", "Channel Data/V2I"]
    dr, lr = Path(drive_root), Path(local_root)
    _t, copied, skipped = time.time(), 0, 0

    for weather in weather_conditions:
        for variant in CHANNEL_VARIANTS:
            cfg_dir = dr / weather / variant / config_name
            if not cfg_dir.exists():
                continue
            for town in towns:
                flat_src = cfg_dir / town          # folder next to Town03.zip
                flat_dst = lr / weather / variant / config_name / town
                if not flat_src.exists() or not any(flat_src.rglob("*_paths.npz")):
                    continue
                for src_file in flat_src.rglob("*_paths.npz"):
                    dst_file = flat_dst / src_file.relative_to(flat_src)
                    if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
                        skipped += 1
                        continue
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    copied += 1

    print(f"Channel flat sync done in {time.time() - _t:.1f}s — "
          f"{copied} copied, {skipped} already cached")
    print(f"  SSD cache: {local_root}")


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
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "verify":
        # Just scan and report — no extraction
        # Usage: !python extraction_drive.py verify
        verify_flat_extraction()

    elif mode == "repair":
        # Check every flat folder; wipe+re-extract any that are bad
        # Usage: !python extraction_drive.py repair
        repair_flat_extraction()

    elif mode == "repair-dry":
        # Preview what repair would do without changing anything
        # Usage: !python extraction_drive.py repair-dry
        repair_flat_extraction(dry_run=True)

    elif mode == "reextract":
        # Verify first, then re-extract only empty folders
        # Usage: !python extraction_drive.py reextract
        empty_ch, empty_sen = verify_flat_extraction()
        if empty_ch or empty_sen:
            print("Re-extracting empty folders …")
            extract_all_to_drive_flat(force_reextract=True)
            print("\nRe-verifying …")
            verify_flat_extraction()
        else:
            print("All folders OK — nothing to re-extract.")

    else:
        # Default: full extract + sync
        # Usage: !python extraction_drive.py
        print("=== Drive flat extraction ===")
        extract_all_to_drive_flat()
        print("\n=== Sync to local SSD ===")
        sync_flat_to_local()
        print(f"\nData ready at: {LOCAL_FLAT}")
