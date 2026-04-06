# Scaled Hierarchical Recovery Notebook Report

Notebook analyzed: `scaled_hiearchical_recovery.ipynb`  
Code-only extracted view: `scaled_hiearchical_recovery_cells_extracted.py` (3178 lines)

## 1) What this notebook is doing overall

This notebook builds and trains a hierarchical federated learning (FL) system for mmWave beam management with four supervised targets:

1. `beam_index` (main classification target)
2. `g_opt` (regression target)
3. `los` (binary classification target)
4. `beam_change` (binary temporal change target)

It runs two training variants:

1. Channel-only (`BeamModel` + `BeamFlowerClient`)
2. Multimodal (`MultimodalBeamModel` + `MultimodalBeamFlowerClient`, CSI + LiDAR + IMU)

It also includes an attacked hierarchical aggregator (`AttackingHierarchicalFedAvg`) and recovery logic in the base strategy (`HierarchicalFedAvg`).

## 2) Cell-by-cell execution map

The practical execution order is:

1. **Cells 0-4 (exploration)**: inspect one ZIP/NPZ and beam distribution. No FL state created.
2. **Cells 6-8**: imports, reproducibility, config (`CFG`), limits (`MAX_CLIENTS`).
3. **Cells 10-12**: dataset classes and sensor alignment classes/functions.
4. **Cell 14**: trajectory grouping, client building, cluster assignment.
5. **Cell 16**: instantiate datasets/clients, compute `beam_changes_global`.
6. **Cell 19**: define model architectures and helper metrics.
7. **Cells 21-24**: define channel-only/multimodal Flower clients and FL strategies.
8. **Cell 25**: run channel-only FL simulation.
9. **Cell 26**: run multimodal FL simulation.
10. **Cells 27-29**: visualize and summarize metrics.
11. **Cell 30**: override beam-change precompute with flat-aware version and optional recompute.

## 3) Data pipeline deep dive

### 3.1 Configuration and reproducibility

From `scaled_hiearchical_recovery_cells_extracted.py`:

1. Lines `130-137` set seeds for Python, NumPy, and TensorFlow, and enforce deterministic TF ops.
2. Lines `144-164` define `CFG`:
   - training params (`local_epochs`, `lr`, `batch_size`, `grad_clip_norm`)
   - codebook sizes (`Q_tx`, `Q_rx`)
   - imbalance controls (`pos_weight_los`, `pos_weight_change`)
   - beam CE regularization (`label_smoothing`)
3. Lines `166-174` set `MAX_CLIENTS` to cap clients for faster experiments.

### 3.2 Dataset indexing and sample resolution

Core class: `ChannelDataset` (starts line `231`).

Key mechanics:

1. `__init__` (`252-311`)
   - resolves weather/config directories across path variants (`CHANNEL_PATH_VARIANTS`).
   - parses antenna counts from config name using `_parse_total_antennas`.
   - builds Tx/Rx DFT codebooks using `generate_dft_codebook`.
   - builds global sample index (`self.index`).

2. `_build_index` (`313-379`)
   - scans either:
     - flat extracted town folders (`TownXX/.../*_paths.npz`) first, or
     - ZIP members (`TownXX.zip`) fallback.
   - stores each sample as `ChannelSampleRef(zip_path, inner_npz)`.
   - applies optional scenario/CAV filters.
   - sorts deterministically by zip + parent path + frame id.

3. `_load_npz` (`416-436`)
   - read priority:
     1. SSD cache (`self.ssd_root`) if present
     2. flat file beside ZIP
     3. ZIP read fallback
   - returns dict of arrays from NPZ.

4. metadata parsing:
   - `_parse_metadata` (`382-388`) extracts `location`, `cav_id`, `frame_id`.
   - `get_sample_metadata` (`390-402`) adds `town`, paths, and `weather`.
   - `build_metadata_index` (`404-410`) builds DataFrame used for trajectory grouping.

### 3.3 Feature extraction from channel

`_extract_csi` (`438-458`) converts raw channel paths into network input:

1. reads `a` tensor from NPZ.
2. squeezes singleton dimensions; ensures shape `(Nr, Nt, n_paths)`.
3. computes effective channel matrix: `H = sum(a_sq over paths)`.
4. builds feature tensor `(Nr, Nt, 3)` as `[real(H), imag(H), abs(H)]`.
5. optional finite-value assertions.

## 4) Label extraction details (non-beam focus)

The exact label dict is created in `ChannelDataset.__getitem__` (`471-507`).

### 4.1 `g_opt` extraction

Source: `compute_beam_index` (`460-469`), called from `__getitem__` line `489`.

Detailed flow:

1. Compute effective channel: `H = sum_paths(a_sq)` (`465`).
2. Compute beamspace response: `response = W_rx^H * H * F_tx` (`466`).
3. Tx-wise max receive gain: `gain_per_tx = max(abs(response)^2 over rx axis)` (`467`).
4. `g_opt = max(gain_per_tx)` (`468`).
5. Save as `np.float32` in label dict (`502`).

Interpretation:

1. `g_opt` is the best achievable beamforming gain for the sample under the current codebooks.
2. It is a channel quality target used by the regression head (`gopt`).

### 4.2 `los` extraction

Source: `__getitem__` lines `491-499`.

Detailed flow:

1. Try direct NPZ key: `arrays.get("los", None)` (`492`).
2. If present, take scalar value from squeezed array (`493-494`).
3. If absent, fallback heuristic:
   - per-path power: `sum(abs(a_sq)^2 over Nr,Nt)` (`496`)
   - total power: `sum(path_power)` (`497`)
   - dominant-path ratio threshold:
     - `los = 1` if `max(path_power)/total_power > 0.85`
     - else `los = 0` (`498`)
4. Save as `np.int32` (`503`).

Interpretation:

1. Prefer dataset-provided LoS flag when available.
2. Use physically motivated fallback when missing.

### 4.3 `beam_change` extraction

`beam_change` is not produced by `ChannelDataset.__getitem__`.  
It is precomputed globally and then looked up by sample index.

Primary precompute implementation: `precompute_beam_changes` (`1064-1177`).

Detailed flow:

1. Build all client-used indices (`1085-1090`).
2. Expand to full relevant trajectories for continuity (`1091-1095`).
3. Attempt cached result (`1098-1111`).
4. Batch read by ZIP group (`1113-1146`), computing beam index per frame via `compute_beam_index`.
5. For each trajectory in frame order:
   - first valid frame gets `0`
   - later frames: `1` if current beam differs from previous, else `0` (`1147-1160`)
6. fill defaults (`1162-1164`), print stats (`1166-1169`), cache save (`1170-1174`).

Override implementation in Cell 30 (`3060-3170`) adds more flat-path candidates and a versioned cache key; it can recompute `beam_changes_global` immediately (`3173-3176`).

## 5) Where each label is consumed in training

### 5.1 Channel-only client path

Class: `BeamFlowerClient` (`1466+`).

1. `_load` (`1490-1535`) reads labels:
   - `beam_index` from dataset labels (`1503`)
   - `g_opt` from dataset labels (`1504`)
   - `los` from dataset labels (`1505`)
   - `beam_change` from global dictionary (`1506`)
2. `g_opt` is z-normalized per client split (`1515-1520`).
3. `fit` (`1550-1621`) losses:
   - beam CE (optionally smoothed) (`1578-1589`)
   - `g_opt` MSE (`1591`)
   - weighted BCE for `los` (`1594-1596`)
   - weighted BCE for `beam_change` (`1597-1599`)
   - weighted sum (`1601-1604`)
4. `evaluate` (`1623-1674`) reports:
   - `gopt_mae`
   - `los_accuracy`
   - `beam_change_accuracy`
   - plus beam metrics and combined loss.

### 5.2 Multimodal client path

Class: `MultimodalBeamFlowerClient` (`2099+`).

Label handling is identical to channel-only:

1. `_load` (`2121-2171`) reads same four targets, with CSI/LiDAR/IMU arrays.
2. `fit` (`2190-2262`) applies same loss components and weights.
3. `evaluate` (`2264-2320`) computes same auxiliary metrics.

## 6) Sensor pipeline (multimodal side)

### 6.1 Sensor index and alignment

`SensorIndex` (`533+`) creates mapping:
`(weather, channel_location, town) -> sensor zip entry`.

Key logic:

1. `_build` (`549-573`) scans sensor zip files by weather/town.
2. `_stem_to_channel_location` (`575-593`) normalizes scenario stem to channel location token.
3. `resolve` (`596-618`) maps each channel inner path to `(entry, cav_id, frame_id)`.

### 6.2 Frame-level sensor extraction

`load_sensor_frame` (`645-703`) does:

1. LiDAR PCD parse via `_read_pcd_xyz` (`625-642`)
2. fixed-size sampling/padding (`676-681`)
3. per-axis normalization (`683-685`)
4. IMU extraction from YAML (`688-701`)
5. returns `{lidar, imu}`.

### 6.3 Multimodal dataset wrapper

`MultimodalChannelDataset` (`709+`):

1. `__getitem__` gets base `csi, labels` from channel dataset (`752`).
2. resolves sensor mapping and loads LiDAR/IMU if available (`754-764`).
3. on missing/failure, returns zero tensors (`765-775`).
4. passes through same label dict from channel side.

## 7) Trajectory -> client -> cluster pipeline

### 7.1 Trajectory creation

`DatasetSplitter` (`845+`) groups metadata by:

1. weather
2. town
3. location
4. cav_id

Then sorts by `frame_id` for each trajectory (`864-870`).

### 7.2 Client construction

`build_clients` (`904-955`) creates one FL client per trajectory:

1. filter short trajectories (`len >= min_trajectory_length`)
2. split each trajectory temporal or random
3. store train/test sample indices in `ChannelClientData`.

### 7.3 Cluster assignment

`build_clusters` (`972-1010`) sorts clients by town/location/weather/cav and partitions evenly into 3 clusters for hierarchical aggregation.

## 8) Strategy behavior and auxiliary-label monitoring

`HierarchicalFedAvg` (`1704+`) uses client-reported stats from `fit`:

1. `mean_g_opt`
2. `mean_los`
3. `mean_beam_change`

These are used in health tracking (`1908-1925`) and in CH re-election scoring (`1773-1794`).

Detection logic combines:

1. cluster loss slope signal
2. CH weight stagnation signal
3. CH divergence signal

Trigger is 2-of-3 voting (`1854-1855`), then expel/isolate/re-entry pipeline.

## 9) Reporting and visualization of non-beam labels

Metrics plotted/reported include:

1. `gopt_mae`
2. `los_accuracy`
3. `beam_change_accuracy`

See plotting blocks in Cells 27-29 (`2736-2844`, `2945-3045`).

## 10) Important implementation notes

1. There are two `precompute_beam_changes` definitions (Cell 16 and Cell 30). The later one overrides the first if executed.
2. `beam_change` semantics differ slightly between versions when trajectory continuity has gaps:
   - first version resets continuity on missing beam (`1155-1157`)
   - override skips non-client indices and defaults missing beam to `0` (`3158-3161`)
3. `g_opt` training target is z-normalized per-client, but strategy health metric returns mean in original scale (`1618`, `2259`).

## 11) Quick reference: most critical code regions

1. Dataset and label generation: lines `231-507`
2. Beam change precompute (first): lines `1064-1177`
3. Channel client label loading + loss: lines `1490-1674`
4. Multimodal client label loading + loss: lines `2121-2320`
5. Hierarchical strategy detection/recovery: lines `1704-2030`
6. Beam change precompute override: lines `3060-3170`

## 12) Bottom line for non-beam labels

1. `g_opt` is physically derived from exhaustive beamspace gain maximization.
2. `los` is direct-from-data when available, else dominant-path ratio heuristic.
3. `beam_change` is trajectory-temporal and computed globally before client training.
4. All three are first-class training targets with explicit losses/metrics and strategy-level monitoring.

