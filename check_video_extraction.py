#!/usr/bin/env python3
"""
检查视频帧提取是否正确。

功能：
1. 读取一个 episode 的一个视角视频
2. 提取帧并保存到指定文件夹
3. 报告视频信息和帧提取状态

Usage:
    python check_video_extraction.py --episode 0
    python check_video_extraction.py --episode 123 --camera observation.images.exterior_image_2_left
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List


DATASET_ROOT = "/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset"


def find_video_path(
    dataset_root: str,
    chunk: str,
    episode: int,
    camera: str = "observation.images.exterior_image_1_left",
) -> str:
    """Find video path for a given episode and camera."""
    filename = f"episode_{episode:06d}.mp4"
    video_path = os.path.join(dataset_root, "videos", chunk, camera, filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path


def extract_frames_ffmpeg(video_path: str, output_dir: Path) -> List[Path]:
    """
    Extract frames from video using FFmpeg directly.
    Saves each frame as a PNG file and returns list of saved paths.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 用 FFmpeg 每帧提取为 PNG
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-fps_mode", "passthrough",  # 保持原始帧率
        f"{output_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    # 收集保存的帧文件
    frame_files = sorted(output_dir.glob("frame_*.png"))
    return frame_files


def main():
    parser = argparse.ArgumentParser(description="检查视频帧提取是否正确")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument("--camera", type=str,
                        default="observation.images.exterior_image_1_left",
                        help="Camera key")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    episode = args.episode
    camera = args.camera
    ep_chunk = episode // 1000
    chunk = f"chunk-{ep_chunk:03d}"
    
    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "debug_frames" / f"ep{episode:06d}_{camera.split('.')[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Episode: {episode}")
    print(f"Camera:  {camera}")
    print(f"Chunk:   {chunk}")
    print(f"Output:  {output_dir}")
    print("=" * 60)
    
    # 1. 定位视频
    try:
        video_path = find_video_path(DATASET_ROOT, chunk, episode, camera)
        print(f"\n[OK] Found video: {video_path}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    
    # 2. 文件信息
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"    File size: {file_size:.2f} MB")
    
    # 3. 提取帧
    print(f"\n[Extracting frames with FFmpeg...]")
    try:
        frame_files = extract_frames_ffmpeg(video_path, output_dir)
        n_frames = len(frame_files)
        print(f"[OK] Extracted {n_frames} frames to {output_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to extract frames: {e}")
        sys.exit(1)
    
    # 4. 获取视频信息
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    probe_info = result.stdout.strip().split(",")
    W, H = int(probe_info[0]), int(probe_info[1])
    fps = eval(probe_info[2]) if probe_info[2] else 0
    
    # 5. 保存元信息
    meta = {
        "episode": episode,
        "camera": camera,
        "chunk": chunk,
        "video_path": video_path,
        "total_frames": n_frames,
        "frame_shape": [H, W, 3],
        "fps": fps,
        "output_dir": str(output_dir),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] Metadata: {meta_path}")
    
    # 6. 总结
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY]")
    print(f"    Video:        {video_path}")
    print(f"    Total frames: {n_frames}")
    print(f"    Resolution:   {W}x{H}")
    print(f"    FPS:          {fps:.2f}")
    print(f"    Output:       {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
