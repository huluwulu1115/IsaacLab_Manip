#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for vision-based policy distillation training.

This module provides helper functions for:
- Video file management and upload to wandb
- Filename parsing and step extraction
- Video file stability checking
"""

import os
import re
import glob
import time
from pathlib import Path
from typing import Optional

import wandb


# ============================================================================
# Video Filename Parsing
# ============================================================================

def extract_step_from_filename(filename: str) -> Optional[int]:
    """Extract step number from video filename.
    
    Args:
        filename: Video filename (e.g., "rl-video-step-10000-xxx.mp4")
    
    Returns:
        Step number extracted from filename, or None if not found
    
    Example:
        >>> extract_step_from_filename("rl-video-step-10000-abc.mp4")
        10000
        >>> extract_step_from_filename("other-file.mp4")
        None
    """
    match = re.search(r'step-(\d+)', filename)
    return int(match.group(1)) if match else None


def get_video_wandb_key(step: int) -> str:
    """Get wandb key for video based on training step.
    
    Videos are organized into two categories:
    - Teacher baseline (step 0): Reference performance
    - Student evaluation (step > 0): Learning progress
    
    Args:
        step: Training step number
    
    Returns:
        Wandb key string ("teacher_baseline/video" or "student_eval/video")
    
    Example:
        >>> get_video_wandb_key(0)
        'teacher_baseline/video'
        >>> get_video_wandb_key(10000)
        'student_eval/video'
    """
    return "teacher_baseline/video" if step == 0 else "student_eval/video"


# ============================================================================
# Video File Stability
# ============================================================================

def wait_for_video_stable(video_dir: Optional[Path], max_wait_time: int = 60) -> bool:
    """Wait for the latest video file to finish writing and become stable.
    
    This function monitors the file size of the most recent video file to determine
    when writing is complete. A file is considered stable when its size remains
    constant for 3 consecutive checks (6 seconds total).
    
    Args:
        video_dir: Directory containing video files
        max_wait_time: Maximum time to wait in seconds (default: 60)
    
    Returns:
        True if video is stable, False if timeout or no videos found
    
    Example:
        >>> video_dir = Path("logs/videos/train")
        >>> wait_for_video_stable(video_dir, max_wait_time=120)
        True  # Video is ready
    """
    if not video_dir or not video_dir.exists():
        return False
    
    all_videos = sorted(glob.glob(str(video_dir / "*.mp4")))
    if not all_videos:
        return False
    
    latest_video = all_videos[-1]
    wait_time = 0
    prev_size = 0
    stable_count = 0
    
    print(f"  ⏳ Waiting for final video to finish writing...")
    
    while wait_time < max_wait_time:
        if not os.path.exists(latest_video):
            time.sleep(2)
            wait_time += 2
            continue
        
        current_size = os.path.getsize(latest_video)
        
        # Check if file size is stable (video writing complete)
        if current_size == prev_size and current_size > 0:
            stable_count += 1
            if stable_count >= 3:  # 3 times stable (6 seconds)
                print(f"  ✓ Final video ready: {os.path.basename(latest_video)} ({current_size/1024:.1f} KB)")
                print(f"  ✓ Wait time: {wait_time}s\n")
                return True
        else:
            stable_count = 0
            prev_size = current_size
            if wait_time % 10 == 0 and wait_time > 0:  # update every 10 seconds
                print(f"     Still writing... ({current_size/1024:.1f} KB, {wait_time}s elapsed)")
        
        time.sleep(2)
        wait_time += 2
    
    print(f"  ⚠️  Timeout waiting for video (waited {max_wait_time}s)")
    return False


# ============================================================================
# Video Upload to Wandb
# ============================================================================

def upload_videos_to_wandb(
    video_dir: Optional[Path],
    uploaded_videos: set,
    fallback_step: Optional[int] = None,
    eval_interval: Optional[int] = None,
    verbose: bool = True
) -> int:
    """Upload new videos to wandb with automatic step detection.
    
    This function discovers new video files, extracts training step from filename,
    and uploads them to the appropriate wandb key. Videos are organized by step:
    - Step 0: Teacher baseline (reference performance)
    - Step > 0: Student evaluation (learning progress)
    
    Args:
        video_dir: Directory containing video files
        uploaded_videos: Set of already uploaded video paths (will be updated in-place)
        fallback_step: Fallback step if filename parsing fails (default: None)
        eval_interval: Evaluation interval for fallback step estimation (default: None)
        verbose: Whether to print upload progress (default: True)
    
    Returns:
        Number of videos successfully uploaded
    
    Example:
        >>> video_dir = Path("logs/videos/train")
        >>> uploaded = set()
        >>> num_uploaded = upload_videos_to_wandb(
        ...     video_dir=video_dir,
        ...     uploaded_videos=uploaded,
        ...     eval_interval=10000
        ... )
        >>> print(f"Uploaded {num_uploaded} videos")
    
    Note:
        The function automatically handles:
        - Filename parsing to extract step number
        - Fallback step calculation if parsing fails
        - Wandb key determination based on step
        - Error handling for failed uploads
    """
    if not video_dir or not video_dir.exists():
        return 0
    
    all_videos = sorted(glob.glob(str(video_dir / "*.mp4")))
    new_videos = [v for v in all_videos if v not in uploaded_videos]
    
    if not new_videos:
        if verbose:
            print(f"  ℹ️  No new videos to upload\n")
        return 0
    
    uploaded_count = 0
    
    for video_path in new_videos:
        video_name = os.path.basename(video_path)
        
        # Extract step from filename
        video_step = extract_step_from_filename(video_name)
        
        if video_step is None:
            # Fallback: use provided fallback or estimate from count
            if fallback_step is not None:
                video_step = fallback_step
            elif eval_interval is not None:
                video_step = len(uploaded_videos) * eval_interval
            else:
                video_step = 0
            
            if verbose:
                print(f"     ⚠️  Could not extract step from {video_name}, using: {video_step}")
        
        # Get wandb key
        wandb_key = get_video_wandb_key(video_step)
        
        if verbose:
            file_size_kb = os.path.getsize(video_path) / 1024
            print(f"     Uploading: {video_name} ({file_size_kb:.1f} KB) → {wandb_key} (step={video_step})")
        
        # Upload to wandb
        try:
            wandb.log({wandb_key: wandb.Video(video_path, fps=30, format="mp4")}, step=video_step)
            uploaded_videos.add(video_path)
            uploaded_count += 1
            if verbose:
                print(f"     ✅ Uploaded")
        except Exception as e:
            if verbose:
                print(f"     ❌ Failed: {e}")
    
    if verbose:
        print(f"  ✓ Uploaded {uploaded_count} video(s)\n")
    
    return uploaded_count

