#!/usr/bin/env python3
"""
Test visualization - Reconstruct frames from atlas and metadata

This script:
1. Reads the atlas PNG and metadata JSON
2. Extracts masks from the atlas using UV coordinates
3. Reconstructs each frame with white-filled parts from metadata and black lines
4. Areas outside parts are transparent
5. Saves reconstructed frames to destination/test_visualize/ folder

Usage:
    python test_visualize.py path/to/folder [destination_path]

Example 1 (default output):
    python test_visualize.py test/Bear
    Output: test/Bear/test_visualize/

Example 2 (custom destination):
    python test_visualize.py test/Bear output
    Output: output/test_visualize/

This will read:
    - test/Bear/Bear.atlas.png
    - test/Bear/Bear.meta.json

And create:
    - [destination]/test_visualize/frame_0000.png (RGBA with transparency)
    - [destination]/test_visualize/frame_0001.png (RGBA with transparency)
    - etc.
"""

import json
import cv2
import numpy as np
import base64
from pathlib import Path
import sys
from tqdm import tqdm


def extract_mask_from_atlas(atlas, uv_coords):
    """
    Extract a mask from the atlas using UV coordinates

    Args:
        atlas: Atlas image (grayscale)
        uv_coords: Dictionary with 'x', 'y', 'width', 'height'

    Returns:
        Extracted mask region
    """
    x = uv_coords['x']
    y = uv_coords['y']
    w = uv_coords['width']
    h = uv_coords['height']

    return atlas[y:y+h, x:x+w].copy()


def reconstruct_frame(atlas, metadata, frame_data):
    """
    Reconstruct a single frame from atlas with white fill for parts only.
    - Outside parts: transparent
    - Inside parts: white fill
    - Black lines: drawn on top
    Uses global_bbox dimensions from metadata as the output size.

    Args:
        atlas: Atlas image (grayscale)
        metadata: Full metadata dictionary
        frame_data: Frame data from metadata['frames'][i]

    Returns:
        Reconstructed frame image (RGBA) with transparent outside, white-filled parts and black lines
    """
    # Get global bbox - this defines the output image size
    global_bbox = metadata['global_bbox']
    bbox_w = global_bbox['width']
    bbox_h = global_bbox['height']

    # Get UV map
    uv_map = metadata['uv_map']

    # Create transparent canvas
    frame_rgba = np.zeros((bbox_h, bbox_w, 4), dtype=np.uint8)

    # Draw white-filled parts from the atlas
    if 'parts' in frame_data:
        for part in frame_data['parts']:
            mask_id = str(part['mask_id'])
            if mask_id not in uv_map:
                continue

            # Get part bbox and mask
            part_bbox = part['bbox']  # [x, y, width, height]
            part_x, part_y, part_w, part_h = part_bbox

            # Extract mask from atlas
            uv_coords = uv_map[mask_id]
            part_mask = extract_mask_from_atlas(atlas, uv_coords)

            # Resize mask if needed to match bbox dimensions
            if part_mask.shape[0] != part_h or part_mask.shape[1] != part_w:
                part_mask = cv2.resize(part_mask, (part_w, part_h), interpolation=cv2.INTER_NEAREST)

            # Draw white fill where mask is white (> 128)
            mask_pixels = part_mask > 128

            # Calculate valid paste region
            canvas_h, canvas_w = frame_rgba.shape[:2]

            # Ensure we don't go out of bounds
            y1 = max(0, part_y)
            y2 = min(canvas_h, part_y + part_h)
            x1 = max(0, part_x)
            x2 = min(canvas_w, part_x + part_w)

            # Calculate corresponding mask region
            mask_y1 = max(0, -part_y)
            mask_y2 = mask_y1 + (y2 - y1)
            mask_x1 = max(0, -part_x)
            mask_x2 = mask_x1 + (x2 - x1)

            if y2 > y1 and x2 > x1:
                region_mask = mask_pixels[mask_y1:mask_y2, mask_x1:mask_x2]
                # Set white color with full opacity
                frame_rgba[y1:y2, x1:x2][region_mask] = [255, 255, 255, 255]

    # Draw black lines on top
    black_mask_id = str(frame_data['black_mask_id'])
    if black_mask_id in uv_map:
        uv_coords = uv_map[black_mask_id]
        black_mask = extract_mask_from_atlas(atlas, uv_coords)

        # Apply black lines: white pixels in mask = black lines in output
        black_pixels = black_mask > 128
        frame_rgba[black_pixels] = [0, 0, 0, 255]  # Black with full opacity

    return frame_rgba


def visualize_from_atlas(folder_path, dest_path=None):
    """
    Reconstruct all frames from atlas and metadata

    Args:
        folder_path: Path to folder containing .atlas.png and .meta.json
        dest_path: Optional destination path. Creates 'test_visualize' subfolder.
                   (default: folder_path/test_visualize)
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # Find atlas and metadata files
    atlas_path = folder_path / f"{folder_name}.atlas.png"
    metadata_path = folder_path / f"{folder_name}.meta.json"

    if not atlas_path.exists():
        print(f"Error: Atlas file not found: {atlas_path}")
        return

    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return

    print(f"Loading atlas from: {atlas_path}")
    atlas = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
    if atlas is None:
        print(f"Error: Could not load atlas image")
        return

    print(f"Atlas size: {atlas.shape[1]}x{atlas.shape[0]}")

    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create output folder
    if dest_path:
        output_dir = Path(dest_path) / folder_name / "test_visualize"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = folder_path / "test_visualize"
        output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Get frame info
    frame_count = len(metadata['frames'])
    global_bbox = metadata['global_bbox']

    # Output size is determined by global_bbox
    img_w = global_bbox['width']
    img_h = global_bbox['height']

    print(f"\nReconstructing {frame_count} frames (parts with white fill, rest transparent)...")
    print(f"Output frame size from global_bbox: {img_w}x{img_h}")

    # Reconstruct each frame
    for frame_idx in tqdm(range(frame_count), desc="Reconstructing frames"):
        frame_data = metadata['frames'][frame_idx]

        # Reconstruct frame
        frame_img = reconstruct_frame(atlas, metadata, frame_data)

        # Save frame
        output_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(output_path), frame_img)

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Created {frame_count} frames in: {output_dir}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python test_visualize.py path/to/folder [destination_path]")
        print("\nExample 1 (default output to folder/test_visualize):")
        print("  python test_visualize.py test/Bear")
        print("  Output: test/Bear/test_visualize/frame_0000.png, ...")
        print("\nExample 2 (custom destination path):")
        print("  python test_visualize.py test/Bear output")
        print("  Output: output/test_visualize/frame_0000.png, ...")
        print("\nThis will read:")
        print("  - test/Bear/Bear.atlas.png")
        print("  - test/Bear/Bear.meta.json")
        print("\nAnd create:")
        print("  - [destination_path]/test_visualize/frame_0000.png")
        print("  - [destination_path]/test_visualize/frame_0001.png")
        print("  - etc.")
        print("\nNote: Reconstructs frames with white-filled parts from atlas, black lines on top, transparent outside")
        sys.exit(1)

    folder_path = sys.argv[1]
    dest_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not Path(folder_path).exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    visualize_from_atlas(folder_path, dest_path)


if __name__ == '__main__':
    main()
