"""
Script to analyze HDF5 files and identify videos with motion issues.
Run this to find which videos trigger "no valid motion; returning zeros"
"""

import h5py
import os
import numpy as np
from glob import glob
from collections import defaultdict


def analyze_landmarks_motion(landmarks):
    """
    Analyze landmarks and return motion status.
    Returns: (status_str, num_valid_frames, num_motion_frames)
    """
    t, f, c = landmarks.shape

    # Check validity (non-zero landmarks)
    valid = ~np.all(landmarks == 0, axis=(1, 2))
    num_valid = np.sum(valid)

    if not np.any(valid):
        return ("ALL_ZERO", 0, 0)

    # Compute motion from valid frames
    motions = np.diff(landmarks[valid], axis=0)
    t_motion = motions.shape[0]

    if t_motion == 0:
        return ("NO_MOTION", num_valid, 0)  # Only 1 valid frame

    # Check if all motions are zero
    if np.allclose(motions, 0):
        return ("ZERO_MOTION", num_valid, t_motion)

    return ("OK", num_valid, t_motion)


def analyze_h5_chunk(chunk_file):
    """Analyze a single HDF5 chunk file."""
    issues = defaultdict(list)

    try:
        with h5py.File(chunk_file, 'r') as h5f:
            video_keys = list(h5f.keys())
            total_videos = len(video_keys)

            for video_key in video_keys:
                try:
                    landmarks = h5f[f"{video_key}/landmarks"][:]
                    status, num_valid, num_motion = analyze_landmarks_motion(landmarks)

                    if status != "OK":
                        issues[status].append({
                            'video_key': video_key,
                            'shape': landmarks.shape,
                            'num_valid': num_valid,
                            'num_motion': num_motion
                        })
                except Exception as e:
                    issues['ERROR'].append({
                        'video_key': video_key,
                        'error': str(e)
                    })

            return total_videos, issues

    except Exception as e:
        print(f"ERROR reading {chunk_file}: {e}")
        return 0, {'CHUNK_ERROR': [{'file': chunk_file, 'error': str(e)}]}


def generate_report(base_path, chunk_pattern="_chunk*.h5"):
    """Generate a comprehensive report for all chunks."""

    # Find all chunk files
    pattern = (base_path if chunk_pattern in base_path
               else os.path.join(base_path, chunk_pattern))
    chunk_files = sorted(glob(pattern))

    if not chunk_files:
        print(f"❌ No HDF5 files found at: {pattern}")
        return

    print(f"\n📊 Analyzing {len(chunk_files)} chunk files...\n")

    # Summary across all chunks
    total_videos = 0
    all_issues = defaultdict(list)
    chunk_summaries = []

    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        num_vids, issues = analyze_h5_chunk(chunk_file)
        total_videos += num_vids

        # Aggregate issues
        for issue_type, video_list in issues.items():
            all_issues[issue_type].extend(video_list)

        # Summary for this chunk
        problematic = sum(len(v) for v in issues.values())
        chunk_summaries.append({
            'name': chunk_name,
            'total': num_vids,
            'problematic': problematic,
            'issues': issues
        })

    # Print detailed report
    print("=" * 80)
    print("LANDMARK & MOTION ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal chunks: {len(chunk_files)}")
    print(f"Total videos: {total_videos}")

    # Per-chunk summary
    print("\n" + "-" * 80)
    print("PER-CHUNK SUMMARY:")
    print("-" * 80)
    for summary in chunk_summaries:
        pct = (summary['problematic'] / summary['total'] * 100) if summary['total'] > 0 else 0
        print(f"\n{summary['name']}")
        print(f"  Total videos: {summary['total']}")
        print(f"  Problematic: {summary['problematic']} ({pct:.1f}%)")

        for issue_type, videos in summary['issues'].items():
            if videos:
                print(f"    {issue_type}: {len(videos)} videos")

    # Detailed issue breakdown
    print("\n" + "-" * 80)
    print("GLOBAL ISSUE BREAKDOWN:")
    print("-" * 80)

    total_issues = sum(len(v) for v in all_issues.values())
    print(f"\nTotal problematic videos: {total_issues} ({total_issues / total_videos * 100:.1f}%)")

    for issue_type, videos in all_issues.items():
        if not videos:
            continue
        print(f"\n🔴 {issue_type.upper()}: {len(videos)} videos")

        if issue_type == "NO_MOTION":
            print("  → Only 1 valid landmark frame (motion = diff → 0 frames)")
            # Show first 5 examples
            for vid_info in videos[:5]:
                print(f"     {vid_info['video_key']}: shape={vid_info['shape']}, valid_frames={vid_info['num_valid']}")
            if len(videos) > 5:
                print(f"     ... and {len(videos) - 5} more")

        elif issue_type == "ALL_ZERO":
            print("  → All landmarks are zero (no valid frames)")
            for vid_info in videos[:5]:
                print(f"     {vid_info['video_key']}: shape={vid_info['shape']}")
            if len(videos) > 5:
                print(f"     ... and {len(videos) - 5} more")

        elif issue_type == "ZERO_MOTION":
            print("  → Motion frames exist but all deltas are zero")
            for vid_info in videos[:5]:
                print(
                    f"     {vid_info['video_key']}: shape={vid_info['shape']}, valid={vid_info['num_valid']}, motion={vid_info['num_motion']}")
            if len(videos) > 5:
                print(f"     ... and {len(videos) - 5} more")

        elif issue_type == "ERROR":
            print("  → Error reading landmarks")
            for vid_info in videos[:5]:
                print(f"     {vid_info['video_key']}: {vid_info['error']}")

    # Export problematic video list
    print("\n" + "=" * 80)
    print("EXPORTING PROBLEMATIC VIDEOS...")
    print("=" * 80)

    problematic_file = os.path.join(os.path.dirname(base_path), "problematic_videos.txt")
    with open(problematic_file, 'w') as f:
        for issue_type, videos in all_issues.items():
            if issue_type != 'CHUNK_ERROR':
                for vid_info in videos:
                    f.write(f"{vid_info['video_key']} ({issue_type})\n")

    print(f"✅ Problematic videos saved to: {problematic_file}")
    print(f"   Total problematic: {total_issues}")

    return all_issues


if __name__ == "__main__":

    base_path = "/home/amin/Projects/Mahsa/datasets/lrs2/lrs2_train_features_chunk*.h5"
    issues = generate_report(base_path)