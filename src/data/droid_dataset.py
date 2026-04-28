"""
DROID-LeRobot dataset loader.

Key conventions:
  - Images are loaded from MP4 video files indexed by parquet metadata.
  - Sampling: every `stride` frames (stride=5 by default).
  - Action = state[t+stride] - state[t]  (state difference with stride).
  - Training: random camera selection each episode.
  - Evaluation: fixed camera (exterior_image_1_left = index 0 in CAMERA_KEYS).
  - Each returned sample: (frames, actions, states, mask_padding, episode_idx).
"""
import json
import os
import struct
from pathlib import Path
from typing import Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm

from ..batch import Batch

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DATASET_ROOT = "/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset"
META_ROOT    = os.path.join(DATASET_ROOT, "meta")
SPLITS_PATH  = "/data3/v-jepa/jepa-wms/1w2splits.json"

IMAGE_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]
STATE_KEY   = "observation.state"
ACTION_KEY  = "action"
TASK_KEYS  = ["observation.state"]  # 8-dim: x,y,z,roll,pitch,yaw,gripper,pad

CAMERA_KEYS = ["observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left"]


def _load_meta():
    """Load episodes.jsonl and info.json."""
    episodes = {}
    with open(os.path.join(META_ROOT, "episodes.jsonl")) as f:
        for line in f:
            ep = json.loads(line)
            episodes[ep["episode_index"]] = ep
    with open(os.path.join(META_ROOT, "info.json")) as f:
        info = json.load(f)
    return episodes, info


def _get_train_val_episodes(
    splits_path: str,
    total_episodes: int,
) -> Tuple[List[int], List[int]]:
    """Split episodes into train/val using the 1w2splits.json file.
    Episodes not in splits["train"] become the validation set."""
    with open(splits_path) as f:
        splits = json.load(f)
    train_ids = set(int(x) for x in splits["train"])
    val_ids   = set(int(x) for x in splits.get("val", [])) | set(int(x) for x in splits.get("test", []))
    val_ids   = sorted(val_ids)
    return sorted(train_ids), val_ids


# ----------------------------------------------------------------------
# Video frame reader (FFmpeg-based, handles AV1)
# ----------------------------------------------------------------------
def _read_mp4_frames_ffmpeg(
    video_path: str,
    frame_indices: List[int],
) -> np.ndarray:
    """
    Read specific frames from an MP4 file using FFmpeg.
    Falls back to returning zeros if FFmpeg fails.
    """
    import subprocess
    
    # Probe for resolution first
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=10)
        probe_info = result.stdout.strip().split(",")
        W, H = int(probe_info[0]), int(probe_info[1])
    except Exception:
        W, H = 640, 480  # fallback default
    
    frame_size = W * H * 3  # rgb24
    total_frames = max(frame_indices) + 1 if frame_indices else 0
    expected_size = frame_size * total_frames
    
    # Extract all frames as raw RGB
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-"
    ]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        all_data = b""
        while True:
            chunk = proc.stdout.read(8192 * 1024)
            if not chunk:
                break
            all_data += chunk
        proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.read().decode())
        
        if not all_data or len(all_data) < frame_size:
            raise RuntimeError("No frame data")
        
        # Reshape to (T, H, W, 3)
        n_read = len(all_data) // frame_size
        frames = np.frombuffer(all_data, dtype=np.uint8).reshape(n_read, H, W, 3)
        
        # Select requested indices
        valid_indices = [i for i in frame_indices if i < n_read]
        if len(valid_indices) == 0:
            return np.zeros((len(frame_indices), H, W, 3), dtype=np.uint8)
        
        result = np.zeros((len(frame_indices), H, W, 3), dtype=np.uint8)
        idx_map = {idx: i for i, idx in enumerate(valid_indices)}
        for new_i, orig_i in enumerate(frame_indices):
            if orig_i in idx_map:
                result[new_i] = frames[idx_map[orig_i]]
        
        return result
        
    except Exception:
        return np.zeros((len(frame_indices), H, W, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# Per-episode data loader
# ----------------------------------------------------------------------
def _load_episode_data(
    episode_index: int,
    camera_key: str,
    stride: int,
) -> dict:
    """Load all frames + states + actions for one episode."""
    ep_chunk   = episode_index // 1000
    ep_file    = f"episode_{episode_index:06d}.parquet"
    parquet_path = os.path.join(DATASET_ROOT, "data", f"chunk-{ep_chunk:03d}", ep_file)
    if not os.path.exists(parquet_path):
        return None
    table = pq.read_table(parquet_path)
    df    = table.to_pandas()

    n_frames = len(df)
    if n_frames < 2:
        return None

    # --- Load states and actions ---
    state_col  = STATE_KEY

    raw_states = np.stack(df[state_col].values).astype(np.float32)   # (N, 8)
    # State has 8 dims: x, y, z, roll, pitch, yaw, pad, gripper
    # Remove the 'pad' dimension (index 6) to get 7-dim state
    states = np.delete(raw_states, 6, axis=1)  # (N, 7)

    # --- Load images ---
    video_path = os.path.join(
        DATASET_ROOT, "videos", f"chunk-{ep_chunk:03d}",
        camera_key, f"episode_{episode_index:06d}.mp4",
    )
    if not os.path.exists(video_path):
        return None

    # Sample every `stride` frames
    sampled_indices = list(range(0, n_frames, stride))
    if len(sampled_indices) < 2:
        return None

    # Load sampled images using FFmpeg
    all_frame_indices = list(range(n_frames))
    img_frames = _read_mp4_frames_ffmpeg(video_path, all_frame_indices)

    # Extract sampled frames
    sampled_frames = img_frames[sampled_indices]  # (N_sampled, H, W, 3)
    sampled_states  = states[sampled_indices]      # (N_sampled, 7)

    # Guard against corrupted/empty frames
    if sampled_frames.shape[0] < 2:
        return None

    # Actions are state differences: action[t] = sampled_state[t+1] - sampled_state[t]
    # (sampled frames are already stride apart in raw time)
    n_sampled = len(sampled_states)
    act_diff = sampled_states[1:] - sampled_states[:-1]  # (N_sampled - 1, 7)

    # Align: each step t uses state[t] and action[t]
    # Number of valid (state, action) pairs = N_sampled - 1 (drop last state)
    n_pairs = min(n_sampled - 1, len(act_diff))

    # Guard against empty / too-short pairs
    if n_pairs <= 0:
        return None

    frames  = sampled_frames[:n_pairs]      # (N_pairs, H, W, 3) uint8
    states_ = sampled_states[:n_pairs]      # (N_pairs, 7)
    acts    = act_diff[:n_pairs]           # (N_pairs, 7) action = state_diff

    return {
        "frames": frames,      # uint8 (N, H, W, 3)
        "states": states_,    # float32 (N, 7)
        "actions": acts,      # float32 (N, 7)
        "episode_index": episode_index,
        "frame_indices": sampled_indices[:n_pairs],
    }


# ----------------------------------------------------------------------
# Torch dataset
# ----------------------------------------------------------------------
class DroidDataset(IterableDataset):
    """
    Yields episodes as (observation_sequence, action_sequence, ...).

    Training mode (mode='train'): random camera each episode.
    Eval mode     (mode='eval'):  fixed camera exterior_image_1_left.

    Each sample is a dict with keys:
      observations : (T, 3, H, W) torch.uint8  [0, 255]
      actions      : (T, act_dim)  torch.float32
      states       : (T, state_dim) torch.float32
      mask_padding : (T,)           torch.bool   (all True for valid frames)
      episode_idx  : int
    """

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        stride: int = 5,
        seq_len: int = 5,
        img_size: int = 256,
        camera_keys: Optional[List[str]] = None,
        num_episodes: Optional[int] = None,
        splits_path: str = SPLITS_PATH,
        dataset_root: str = DATASET_ROOT,
        preload: bool = False,
    ):
        self.split       = split
        self.stride      = stride
        self.seq_len     = seq_len
        self.img_size    = img_size
        self.num_episodes_limit = num_episodes
        self.dataset_root = dataset_root

        # Load metadata
        self.episodes, self.info = _load_meta()
        total_eps = self.info["total_episodes"]
        train_ids, val_ids = _get_train_val_episodes(splits_path, total_eps)
        self.episode_ids = train_ids if split == "train" else val_ids

        if num_episodes is not None:
            self.episode_ids = self.episode_ids[:num_episodes]

        self.all_cameras = camera_keys or CAMERA_KEYS
        self.state_dim  = 7  # x,y,z,roll,pitch,yaw,gripper
        self.action_dim = 7

        # Pre-load all episodes into memory for speed
        self._preloaded = preload
        self._cache: dict = {}
        if preload:
            print(f"[DroidDataset] Preloading {len(self.episode_ids)} episodes ({split})...")
            for ep_id in tqdm(self.episode_ids):
                self._preload_episode(ep_id)

    def _preload_episode(self, ep_id: int) -> Optional[dict]:
        cam_key = self._get_camera(ep_id)
        cache_key = (ep_id, cam_key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        data = _load_episode_data(ep_id, cam_key, self.stride)
        if data is not None:
            self._cache[cache_key] = data
        return data

    def _get_camera(self, ep_id: int) -> str:
        if self.split == "eval":
            return self.all_cameras[0]  # fixed: exterior_image_1_left
        import random
        return random.choice(self.all_cameras)

    def __len__(self):
        return len(self.episode_ids)

    def __iter__(self):
        import random
        worker_info = get_worker_info()
        if worker_info is not None:
            ep_ids = self.episode_ids[worker_info.id :: worker_info.num_workers]
        else:
            ep_ids = list(self.episode_ids)

        # Shuffle episode order each epoch for training
        if self.split == "train":
            random.shuffle(ep_ids)

        for ep_id in ep_ids:
            sample = self._fetch_episode(ep_id)
            if sample is None:
                continue
            yield sample

    def _fetch_episode(self, ep_id: int) -> Optional[dict]:
        cam_key = self._get_camera(ep_id)
        cache_key = (ep_id, cam_key)

        if self._preloaded:
            data = self._cache.get(cache_key)
            if data is None:
                data = _load_episode_data(ep_id, cam_key, self.stride)
                if data is not None:
                    self._cache[cache_key] = data
        else:
            data = _load_episode_data(ep_id, cam_key, self.stride)

        if data is None:
            return None

        frames   = data["frames"]    # (N, H, W, 3) uint8
        states   = data["states"]     # (N, 7) float32
        actions  = data["actions"]    # (N, 7) float32
        frm_idxs = data["frame_indices"]

        N = len(frames)
        if N < self.seq_len:
            return None

        # Random contiguous subsequence of length seq_len
        import random
        start = random.randint(0, N - self.seq_len)
        end   = start + self.seq_len

        # Resize frames to img_size (using PIL for reliability)
        try:
            from PIL import Image
            frames_crop = []
            for f in frames[start:end]:
                img = Image.fromarray(f)
                resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                frames_crop.append(np.array(resized))
            frames_crop = np.stack(frames_crop, axis=0)  # (T, H, W, 3)
        except ImportError:
            # Fallback: bilinear resize with torch
            f_t = torch.from_numpy(frames[start:end]).float().permute(0, 3, 1, 2) / 255.0
            f_t = F.interpolate(f_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            frames_crop = (f_t * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

        # No normalization for states/actions - use raw values
        # Action is state difference: action[t] = state[t+stride] - state[t]
        states_np = np.array(states[start:end], dtype=np.float32)
        actions_np = np.array(actions[start:end], dtype=np.float32)

        return {
            "observations": torch.from_numpy(frames_crop).permute(0, 3, 1, 2).byte(),   # (T, 3, H, W)
            "actions":     torch.from_numpy(actions_np).float(),
            "states":      torch.from_numpy(states_np).float(),
            "mask_padding": torch.ones(self.seq_len, dtype=torch.bool),
            "episode_idx":  ep_id,
            "frame_idx":   torch.tensor(frm_idxs[start:end]),
        }

    def collate_fn(self, batch: List[dict]) -> Batch:
        """Simple collate: stack all tensors."""
        observations  = torch.stack([b["observations"]  for b in batch])
        actions       = torch.stack([b["actions"]      for b in batch])
        states        = torch.stack([b["states"]       for b in batch])
        mask_padding  = torch.stack([b["mask_padding"] for b in batch])
        episode_idx   = torch.tensor([b["episode_idx"]  for b in batch])
        frame_idx     = torch.stack([b["frame_idx"]    for b in batch])
        return Batch(
            observations=observations,
            actions=actions,
            states=states,
            mask_padding=mask_padding,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
        )


def build_dataloaders(
    stride: int = 5,
    seq_len: int = 5,
    img_size: int = 64,
    batch_size: int = 16,
    num_workers: int = 4,
    train_episodes: Optional[int] = None,
    val_episodes: Optional[int] = None,
    preload: bool = False,
):
    train_ds = DroidDataset(
        split="train", stride=stride, seq_len=seq_len, img_size=img_size,
        preload=preload, num_episodes=train_episodes,
    )
    val_ds = DroidDataset(
        split="val", stride=stride, seq_len=seq_len, img_size=img_size,
        preload=preload, num_episodes=val_episodes,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=train_ds.collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=val_ds.collate_fn, pin_memory=True,
    )
    return train_loader, val_loader
