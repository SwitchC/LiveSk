#!/usr/bin/env python3
"""
Test main image generation - Validate main.png against atlas reconstruction

This script:
1. Reads folder_name.main.png and folder_name.meta.json
2. Reconstructs the main frame from atlas using main_frame_id
3. Gets black lines bbox from the main frame
4. Creates new image with global_bbox dimensions
5. Positions main.png at the black lines bbox location
6. Compares the positioned image with the reconstructed frame
7. Saves all images to destination/test_main_image/ folder:
   - new_image.png (main.png positioned in global bbox)
   - main_frame_id.png (reconstructed from atlas)
   - diff.png (difference - should be fully white if correct)

Usage:
    python test_main_image.py path/to/folder [destination_path]

Example 1 (default output):
    python test_main_image.py colored/Bear
    Output: colored/Bear/test_main_image/

Example 2 (custom destination):
    python test_main_image.py colored/Bear output
    Output: output/test_main_image/

This will read:
    - colored/Bear/Bear.main.png
    - colored/Bear/Bear.atlas.png
    - colored/Bear/Bear.meta.json

And create:
    - [destination]/test_main_image/new_image.png
    - [destination]/test_main_image/main_frame_id.png
    - [destination]/test_main_image/diff.png
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path


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


def get_black_lines_bbox(atlas, metadata, frame_idx):
    """
    Get the bounding box of black lines for a specific frame

    Args:
        atlas: Atlas image (grayscale)
        metadata: Metadata dictionary
        frame_idx: Frame index

    Returns:
        Dictionary with 'x', 'y', 'width', 'height' or None if no content
    """
    frame_data = metadata['frames'][frame_idx]
    uv_map = metadata['uv_map']

    # Get black lines mask
    black_mask_id = str(frame_data['black_mask_id'])
    if black_mask_id not in uv_map:
        return None

    uv_coords = uv_map[black_mask_id]
    black_mask = extract_mask_from_atlas(atlas, uv_coords)

    # Find bounding box of black pixels
    points = np.where(black_mask > 128)
    if len(points[0]) == 0:
        return None

    y_min, y_max = points[0].min(), points[0].max()
    x_min, x_max = points[1].min(), points[1].max()

    return {
        'x': int(x_min),
        'y': int(y_min),
        'width': int(x_max - x_min + 1),
        'height': int(y_max - y_min + 1)
    }


def reconstruct_frame_from_atlas(atlas, metadata, frame_idx):
    """
    Reconstruct a frame from atlas (black lines only)

    Args:
        atlas: Atlas image (grayscale)
        metadata: Metadata dictionary
        frame_idx: Frame index

    Returns:
        Reconstructed frame with black lines on white background (BGR)
    """
    global_bbox = metadata['global_bbox']
    bbox_w = global_bbox['width']
    bbox_h = global_bbox['height']

    # Create white background
    frame = np.ones((bbox_h, bbox_w, 3), dtype=np.uint8) * 255

    # Get frame data
    frame_data = metadata['frames'][frame_idx]
    uv_map = metadata['uv_map']

    # Draw black lines
    black_mask_id = str(frame_data['black_mask_id'])
    if black_mask_id in uv_map:
        uv_coords = uv_map[black_mask_id]
        black_mask = extract_mask_from_atlas(atlas, uv_coords)

        # Apply black lines: white pixels in mask = black lines in output
        black_pixels = black_mask > 128
        frame[black_pixels] = [0, 0, 0]  # Black

    return frame


def test_main_image(folder_path, dest_path=None):
    """
    Test main image generation

    Args:
        folder_path: Path to folder containing main.png and metadata
        dest_path: Optional destination path. Creates 'test_main_image' subfolder.
                   (default: folder_path/test_main_image)
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # Define file paths
    main_png_path = folder_path / f"{folder_name}.main.png"
    atlas_path = folder_path / f"{folder_name}.atlas.png"
    metadata_path = folder_path / f"{folder_name}.meta.json"

    # Check if files exist
    if not main_png_path.exists():
        print(f"Error: Main image not found: {main_png_path}")
        return

    if not atlas_path.exists():
        print(f"Error: Atlas not found: {atlas_path}")
        return

    if not metadata_path.exists():
        print(f"Error: Metadata not found: {metadata_path}")
        return

    print(f"Loading files from: {folder_path}")
    print(f"  Main image: {main_png_path}")
    print(f"  Atlas:      {atlas_path}")
    print(f"  Metadata:   {metadata_path}")

    # Load main.png (RGBA with transparency)
    main_img = cv2.imread(str(main_png_path), cv2.IMREAD_UNCHANGED)
    if main_img is None:
        print(f"Error: Could not load main image")
        return

    print(f"\nMain image size: {main_img.shape[1]}x{main_img.shape[0]}")
    if main_img.shape[2] == 4:
        print("  Format: RGBA (with transparency)")
    else:
        print("  Format: BGR")

    # Load atlas
    atlas = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
    if atlas is None:
        print(f"Error: Could not load atlas")
        return

    print(f"Atlas size: {atlas.shape[1]}x{atlas.shape[0]}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    main_frame_id = metadata.get('main_frame_id', 0)
    global_bbox = metadata['global_bbox']

    print(f"\nMetadata:")
    print(f"  Main frame ID: {main_frame_id}")
    print(f"  Global bbox: {global_bbox['width']}x{global_bbox['height']}")

    # Get black lines bbox
    black_bbox = get_black_lines_bbox(atlas, metadata, main_frame_id)
    if black_bbox is None:
        print("Error: Could not find black lines bbox")
        return

    print(f"  Black lines bbox: {black_bbox['width']}x{black_bbox['height']} at ({black_bbox['x']}, {black_bbox['y']})")

    # Step 1: Create new image with global_bbox dimensions (white background)
    new_image = np.ones((global_bbox['height'], global_bbox['width'], 3), dtype=np.uint8) * 255

    # Step 2: Convert main.png to BGR if it's RGBA
    if main_img.shape[2] == 4:
        # Create white background
        main_bgr = np.ones((main_img.shape[0], main_img.shape[1], 3), dtype=np.uint8) * 255
        # Composite RGBA over white background
        alpha = main_img[:, :, 3:4] / 255.0
        main_bgr = main_img[:, :, :3] * alpha + main_bgr * (1 - alpha)
        main_bgr = main_bgr.astype(np.uint8)
    else:
        main_bgr = main_img

    # Step 3: Position main.png at black_bbox location in new_image
    paste_x = black_bbox['x']
    paste_y = black_bbox['y']

    main_h, main_w = main_bgr.shape[:2]
    new_h, new_w = new_image.shape[:2]

    # Calculate valid paste region
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(main_w, new_w - paste_x)
    src_y2 = min(main_h, new_h - paste_y)

    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    dst_x2 = min(new_w, paste_x + main_w)
    dst_y2 = min(new_h, paste_y + main_h)

    # Paste main image
    if src_x2 > src_x1 and src_y2 > src_y1:
        new_image[dst_y1:dst_y2, dst_x1:dst_x2] = main_bgr[src_y1:src_y2, src_x1:src_x2]

    print(f"\nPositioned main.png at ({paste_x}, {paste_y}) in global bbox")

    # Step 4: Reconstruct frame from atlas
    main_frame_reconstructed = reconstruct_frame_from_atlas(atlas, metadata, main_frame_id)

    print(f"Reconstructed main frame from atlas")

    # Step 5: Create diff image
    diff = cv2.absdiff(new_image, main_frame_reconstructed)

    # Check if diff is all white (perfect match)
    is_white = np.all(diff == 0)
    if is_white:
        print("\n✓ SUCCESS: Diff image is fully white - perfect match!")
    else:
        non_white_pixels = np.sum(diff > 0)
        total_pixels = diff.shape[0] * diff.shape[1] * diff.shape[2]
        percentage = (non_white_pixels / total_pixels) * 100
        print(f"\n✗ DIFFERENCE FOUND: {non_white_pixels}/{total_pixels} pixels differ ({percentage:.2f}%)")

    # Create output folder
    if dest_path:
        output_dir = Path(dest_path) / folder_name / "test_main_image"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = folder_path / "test_main_image"
        output_dir.mkdir(exist_ok=True)

    # Save images
    new_image_path = output_dir / "new_image.png"
    main_frame_path = output_dir / "main_frame_id.png"
    diff_path = output_dir / "diff.png"

    cv2.imwrite(str(new_image_path), new_image)
    cv2.imwrite(str(main_frame_path), main_frame_reconstructed)
    cv2.imwrite(str(diff_path), diff)

    print(f"\nSaved output images to: {output_dir}")
    print(f"  new_image.png:     {new_image_path}")
    print(f"  main_frame_id.png: {main_frame_path}")
    print(f"  diff.png:          {diff_path}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python test_main_image.py path/to/folder [destination_path]")
        print("\nExample 1 (default output to folder/test_main_image):")
        print("  python test_main_image.py colored/Bear")
        print("  Output: colored/Bear/test_main_image/new_image.png, ...")
        print("\nExample 2 (custom destination path):")
        print("  python test_main_image.py colored/Bear output")
        print("  Output: output/test_main_image/new_image.png, ...")
        print("\nThis will read:")
        print("  - colored/Bear/Bear.main.png")
        print("  - colored/Bear/Bear.atlas.png")
        print("  - colored/Bear/Bear.meta.json")
        print("\nAnd create:")
        print("  - [destination_path]/test_main_image/new_image.png")
        print("  - [destination_path]/test_main_image/main_frame_id.png")
        print("  - [destination_path]/test_main_image/diff.png")
        sys.exit(1)

    folder_path = sys.argv[1]
    dest_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not Path(folder_path).exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    test_main_image(folder_path, dest_path)


if __name__ == '__main__':
    main()
