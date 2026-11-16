#!/usr/bin/env python3
"""
Prepare data for animation - process frames and create atlas + main image

This script:
1. Takes a folder path and frame number as input
2. Reads images from path_to_folder/final
3. Processes frames to extract masks and color data
4. Creates resized images in path_to_folder/final_resize
5. Creates atlas and metadata
6. Generates main image from the atlas using specified frame number
7. Outputs:
   - folder_name.atlas.png (texture atlas)
   - folder_name.meta.json (metadata with bbox and mask info)
   - folder_name.main.png (main image with transparent background)

Usage:
    python prepare_data.py path/to/folder frame_number

Example:
    python prepare_data.py colored/Bear 0
    (Uses frame 0 from the atlas to generate main image)
"""

import cv2
import numpy as np
import json
import sys
import base64
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
from constants import (
    RACCOON_ETALON,
    BLACK_THRESHOLD,
    WHITE_THRESHOLD,
    BBOX_PADDING,
    MAX_ATLAS_WIDTH,
    OUTPUT_IMAGE_SIZE
)


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def get_color_tuple(color):
    """Normalize color to tuple for consistent comparison"""
    return tuple(map(int, color))


def is_white_color(color):
    """Check if a color is white or near-white based on WHITE_THRESHOLD"""
    return all(c >= WHITE_THRESHOLD for c in color)


def extract_black_lines(image):
    """Extract black lines/outlines from the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return black_mask


def create_transparent_background(image):
    """
    Convert image to RGBA with transparent background around black lines.
    Black lines become black with full opacity, everything else becomes transparent.

    Args:
        image: BGR image (from cv2.imread)

    Returns:
        RGBA image with transparent background
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = black_mask  # Set alpha channel

    return bgra


def find_content_bounding_box(image):
    """
    Find bounding box of non-white content in the image.

    Args:
        image: BGR image

    Returns:
        dict with 'x', 'y', 'width', 'height' or None if no content found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    points = np.where(binary > 0)
    if len(points[0]) == 0:
        return None

    y_min, y_max = points[0].min(), points[0].max()
    x_min, x_max = points[1].min(), points[1].max()

    # Add padding
    h, w = image.shape[:2]
    x_min = max(0, x_min - BBOX_PADDING)
    y_min = max(0, y_min - BBOX_PADDING)
    x_max = min(w - 1, x_max + BBOX_PADDING)
    y_max = min(h - 1, y_max + BBOX_PADDING)

    return {
        'x': int(x_min),
        'y': int(y_min),
        'width': int(x_max - x_min + 1),
        'height': int(y_max - y_min + 1)
    }


def encode_mask(mask):
    """Encode binary mask to base64 string"""
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')


def get_mask_hash(mask):
    """Generate a hash for a mask to identify duplicates"""
    mask_bytes = mask.tobytes()
    return hashlib.md5(mask_bytes).hexdigest()


# ============================================================================
# COLOR AND REGION DETECTION
# ============================================================================

def parse_colors_from_frame(image, black_mask):
    """Parse unique colors from frame, excluding black lines and white background"""
    h, w = image.shape[:2]
    unique_colors = set()

    # Create mask for non-black areas
    color_mask = (black_mask == 0)

    # Vectorized approach for speed
    pixels = image.reshape(-1, 3)
    mask_flat = color_mask.reshape(-1)

    # Filter out black lines
    valid_pixels = pixels[mask_flat]

    # Filter out white/near-white pixels (>= WHITE_THRESHOLD to include exact threshold value)
    non_white_mask = ~np.all(valid_pixels >= WHITE_THRESHOLD, axis=1)
    valid_colors = valid_pixels[non_white_mask]

    # Get unique colors
    if len(valid_colors) > 0:
        unique_colors_array = np.unique(valid_colors, axis=0)
        unique_colors = {tuple(color) for color in unique_colors_array}

    return unique_colors


# ============================================================================
# BOUNDING BOX CALCULATION
# ============================================================================

def find_global_bounding_box(image_paths):
    """Find the maximum bounding box that encompasses all colored parts across all frames"""
    print("Finding global bounding box across all frames...")

    global_min_x = float('inf')
    global_min_y = float('inf')
    global_max_x = 0
    global_max_y = 0

    for idx, image_path in enumerate(tqdm(image_paths, desc="Finding global bbox")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # Extract black lines
        black_mask = extract_black_lines(image)

        # Create mask for all colored (non-black, non-white) areas
        color_mask = (black_mask == 0).astype(np.uint8)

        # Remove white background
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_white_mask = (gray < WHITE_THRESHOLD).astype(np.uint8)

        # Combined mask: colored areas only
        combined_mask = cv2.bitwise_and(color_mask, non_white_mask)

        # Find all non-zero points
        points = np.where(combined_mask > 0)

        if len(points[0]) > 0:
            y_min, y_max = points[0].min(), points[0].max()
            x_min, x_max = points[1].min(), points[1].max()

            global_min_x = min(global_min_x, x_min)
            global_min_y = min(global_min_y, y_min)
            global_max_x = max(global_max_x, x_max)
            global_max_y = max(global_max_y, y_max)

    # Calculate global bounding box with padding
    global_x = max(0, int(global_min_x) - BBOX_PADDING)
    global_y = max(0, int(global_min_y) - BBOX_PADDING)
    global_w = int(global_max_x - global_min_x + 1) + 2 * BBOX_PADDING
    global_h = int(global_max_y - global_min_y + 1) + 2 * BBOX_PADDING

    print(f"\nGlobal bounding box: x={global_x}, y={global_y}, w={global_w}, h={global_h}")

    return {
        'x': global_x,
        'y': global_y,
        'width': global_w,
        'height': global_h
    }


def calculate_smart_scale_to_etalon(global_bbox):
    """
    Calculate scale factor to fit current creature's bbox into Raccoon's bbox

    Args:
        global_bbox: dict with 'x', 'y', 'width', 'height' of current creature

    Returns:
        scale_factor: float to resize the creature to fit in Raccoon's bbox
    """
    bbox_w = global_bbox['width']
    bbox_h = global_bbox['height']

    # Get Raccoon etalon bbox
    etalon_bbox_w = RACCOON_ETALON['global_bbox']['width']
    etalon_bbox_h = RACCOON_ETALON['global_bbox']['height']

    # Calculate scale to fit current bbox into Raccoon's bbox
    scale_x = etalon_bbox_w / bbox_w
    scale_y = etalon_bbox_h / bbox_h

    # Use the smaller scale to ensure bbox fits completely
    scale_factor = min(scale_x, scale_y)

    print(f"Smart scale calculation (fit into Raccoon bbox):")
    print(f"  Current bbox: {bbox_w}x{bbox_h}")
    print(f"  Raccoon bbox: {etalon_bbox_w}x{etalon_bbox_h}")
    print(f"  Calculated scale factor: {scale_factor:.4f}")
    print(f"  Scaled bbox will be: {int(bbox_w * scale_factor)}x{int(bbox_h * scale_factor)}")

    return scale_factor


# ============================================================================
# MASK AND REGION PROCESSING
# ============================================================================

def find_color_regions(image_cropped, color, black_mask_cropped):
    """Find all regions with a specific color and merge them into a single mask"""
    # Skip white colors
    if is_white_color(color):
        return []

    h, w = image_cropped.shape[:2]

    # Create mask for this specific color
    color_array = np.array(color, dtype=np.uint8)

    # Fast comparison using numpy broadcasting
    color_matches = np.all(image_cropped == color_array, axis=2)
    color_mask = (color_matches & (black_mask_cropped == 0)).astype(np.uint8) * 255

    if not np.any(color_mask):
        return []

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Find all non-zero points in the merged mask
    y_coords, x_coords = np.where(color_mask > 0)

    if len(x_coords) <= 10:  # Filter out noise
        return []

    # Calculate bounding box for the entire merged mask (encompasses all regions)
    x_min = int(x_coords.min())
    y_min = int(y_coords.min())
    x_max = int(x_coords.max())
    y_max = int(y_coords.max())

    bbox_w = int(x_max - x_min + 1)
    bbox_h = int(y_max - y_min + 1)

    # Calculate area (total pixels with this color, all regions combined)
    area = int(np.sum(color_mask > 0))

    # Calculate centroid for the merged mask (weighted center of all regions)
    M = cv2.moments(color_mask)
    if M["m00"] != 0:
        cx_abs = int(M["m10"] / M["m00"])
        cy_abs = int(M["m01"] / M["m00"])
    else:
        cx_abs = int(x_min + bbox_w // 2)
        cy_abs = int(y_min + bbox_h // 2)

    # Centroid relative to part's bbox
    cx_rel = int(cx_abs - x_min)
    cy_rel = int(cy_abs - y_min)

    # Extract cropped mask for the merged region
    cropped_mask = color_mask[y_min:y_max + 1, x_min:x_max + 1]

    # Return a single merged region
    return [{
        'bbox': [x_min, y_min, bbox_w, bbox_h],
        'centroid_global': [cx_abs, cy_abs],
        'centroid_rel': [cx_rel, cy_rel],
        'mask': cropped_mask,
        'area': area
    }]


# ============================================================================
# MASK LIBRARY (DEDUPLICATION)
# ============================================================================

class MaskLibrary:
    """Manages unique masks and their IDs"""

    def __init__(self):
        self.masks = {}  # hash -> (mask_id, encoded_mask, dimensions)
        self.next_id = 1
        self.hash_to_id = {}

    def add_mask(self, mask, bbox):
        """Add a mask to the library and return its ID"""
        # Generate hash for the mask
        mask_hash = get_mask_hash(mask)

        # Check if this mask already exists
        if mask_hash in self.hash_to_id:
            return self.hash_to_id[mask_hash]

        # New unique mask
        mask_id = self.next_id
        self.next_id += 1

        # Store the mask
        encoded_mask = encode_mask(mask)
        self.masks[mask_id] = {
            'id': mask_id,
            'data': encoded_mask,
            'dimensions': [bbox[2], bbox[3]],  # width, height
            'hash': mask_hash
        }
        self.hash_to_id[mask_hash] = mask_id

        return mask_id

    def get_library_data(self):
        """Get all masks for storage"""
        return {
            'count': len(self.masks),
            'masks': list(self.masks.values())
        }

    def get_statistics(self):
        """Get statistics about mask deduplication"""
        return {
            'unique_masks': len(self.masks),
            'total_size': sum(len(m['data']) for m in self.masks.values())
        }


# ============================================================================
# FRAME PROCESSING
# ============================================================================

def process_frame_with_mask_library(image, black_mask, frame_idx, color_to_id, global_bbox, global_bbox_original, mask_library):
    """Process a frame and add unique masks to the library"""
    h, w = image.shape[:2]

    # Extract the global bbox region using original bbox
    gx, gy, gw, gh = global_bbox_original['x'], global_bbox_original['y'], global_bbox_original['width'], global_bbox_original['height']

    # Ensure bbox is within image bounds
    x_start = max(0, gx)
    y_start = max(0, gy)
    x_end = min(w, gx + gw)
    y_end = min(h, gy + gh)

    # Create cropped versions relative to global bbox
    image_cropped = np.ones((gh, gw, 3), dtype=np.uint8) * 255  # White background
    black_mask_cropped = np.zeros((gh, gw), dtype=np.uint8)

    # Calculate offsets for placing the image content
    paste_x = max(0, -gx)
    paste_y = max(0, -gy)

    # Copy image regions to cropped versions
    crop_h = y_end - y_start
    crop_w = x_end - x_start

    if crop_h > 0 and crop_w > 0:
        image_cropped[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = image[y_start:y_end, x_start:x_end]
        black_mask_cropped[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = black_mask[y_start:y_end,
                                                                                 x_start:x_end]

    # Process black lines mask
    black_mask_id = mask_library.add_mask(black_mask_cropped, [0, 0, gw, gh])

    # Process colored regions
    frame_parts = []

    for color, color_id in color_to_id.items():
        regions = find_color_regions(image_cropped, color, black_mask_cropped)

        for region in regions:
            # Add mask to library and get its ID
            mask_id = mask_library.add_mask(region['mask'], region['bbox'])

            # Create part data with mask_id instead of mask data
            part_data = {
                'id': int(color_id),
                'color': [int(c) for c in color],  # BGR format
                'bbox': region['bbox'],  # Relative to global bbox
                'centroid_global': region['centroid_global'],  # Relative to global bbox
                'centroid_rel': region['centroid_rel'],  # Relative to part's bbox
                'area': region['area'],
                'mask_id': mask_id  # Reference to mask in library
            }
            frame_parts.append(part_data)

    # Sort parts by ID for consistency
    frame_parts.sort(key=lambda x: (x['id'], -x['area']))

    return {
        'frame_index': int(frame_idx),
        'black_lines_mask_id': black_mask_id,  # Reference to black lines mask
        'parts_count': len(frame_parts),
        'parts': frame_parts
    }


def process_frames_with_deduplication(image_paths):
    """Process all frames with mask deduplication"""
    if not image_paths:
        return None

    # Initialize mask library
    mask_library = MaskLibrary()

    # Step 1: Find global bounding box
    global_bbox_original = find_global_bounding_box(image_paths)

    # Store original offset
    bbox_offset_x = global_bbox_original['x']
    bbox_offset_y = global_bbox_original['y']

    # Create adjusted bbox starting from 0,0
    global_bbox = {
        'x': 0,
        'y': 0,
        'width': global_bbox_original['width'],
        'height': global_bbox_original['height']
    }

    # Step 2: Establish color mapping from first frame
    print("\nEstablishing color mapping from first frame...")
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        return None

    black_mask = extract_black_lines(first_image)
    frame_colors = parse_colors_from_frame(first_image, black_mask)

    color_to_id = {}
    next_id = 1
    for color in sorted(frame_colors):
        color_to_id[color] = next_id
        print(f"  Color {color} -> ID {next_id}")
        next_id += 1

    # Step 3: Process all frames with mask deduplication
    print(f"\nProcessing {len(image_paths)} frames with mask deduplication...")
    all_frames_data = []
    total_mask_references = 0

    for frame_idx, image_path in enumerate(tqdm(image_paths, desc="Processing frames")):
        image = cv2.imread(str(image_path))
        if image is None:
            tqdm.write(f"  Warning: Could not read {image_path}")
            continue

        # Extract black lines
        black_mask = extract_black_lines(image)

        # Check for new colors
        frame_colors = parse_colors_from_frame(image, black_mask)
        new_colors = frame_colors - set(color_to_id.keys())

        if new_colors:
            for color in sorted(new_colors):
                color_to_id[color] = next_id
                next_id += 1

        # Process frame with mask library
        frame_data = process_frame_with_mask_library(
            image, black_mask, frame_idx, color_to_id, global_bbox, global_bbox_original, mask_library
        )
        all_frames_data.append(frame_data)
        total_mask_references += frame_data['parts_count'] + 1  # parts + black lines

    # Get mask library statistics
    mask_stats = mask_library.get_statistics()

    print("\n" + "=" * 50)
    print("Mask Deduplication Statistics:")
    print(f"  Total mask references: {total_mask_references}")
    print(f"  Unique masks stored: {mask_stats['unique_masks']}")
    print(f"  Deduplication ratio: {total_mask_references / mask_stats['unique_masks']:.2f}x")
    print(f"  Space saved: {(1 - mask_stats['unique_masks'] / total_mask_references) * 100:.1f}%")

    return {
        'global_bbox': global_bbox,
        'frame_count': len(all_frames_data),
        'color_mapping': {f"{color[0]},{color[1]},{color[2]}": int(id_val)
                          for color, id_val in sorted(color_to_id.items(), key=lambda x: x[1])},
        'total_unique_ids': len(color_to_id),
        'mask_library': mask_library.get_library_data(),
        'frames': all_frames_data
    }


# ============================================================================
# TEXTURE ATLAS PACKING
# ============================================================================

class TexturePacker:
    """Simple shelf-based texture packing algorithm"""

    def __init__(self, max_width=MAX_ATLAS_WIDTH):
        self.max_width = max_width
        self.rects = []

    def pack(self, masks):
        """Pack masks and return atlas dimensions and positions"""
        # Sort by height (descending) for better packing
        sorted_masks = sorted(masks, key=lambda m: m['height'], reverse=True)

        current_y = 0
        current_x = 0
        row_height = 0
        max_width_used = 0

        for mask in sorted_masks:
            w, h = mask['width'], mask['height']

            # Check if we need a new row
            if current_x + w > self.max_width:
                current_y += row_height
                current_x = 0
                row_height = 0

            # Place mask
            self.rects.append({
                'mask_id': mask['id'],
                'x': current_x,
                'y': current_y,
                'width': w,
                'height': h,
                'mask_data': mask
            })

            current_x += w
            row_height = max(row_height, h)
            max_width_used = max(max_width_used, current_x)

        # Final atlas dimensions
        atlas_width = max_width_used
        atlas_height = current_y + row_height

        return atlas_width, atlas_height


# ============================================================================
# ATLAS AND METADATA CREATION
# ============================================================================

def create_atlas_and_metadata(data, output_prefix, main_frame_id=0):
    """Create PNG atlas and metadata JSON from processed data

    Args:
        data: Processed frame data
        output_prefix: Output file path prefix
        main_frame_id: Frame index used for main image
    """

    print("\nCreating atlas and metadata...")

    # Prepare masks
    masks_list = data['mask_library']['masks']

    print(f"Processing {len(masks_list)} masks...")
    masks_to_pack = []

    # Decode masks for atlas
    for mask_data in tqdm(masks_list, desc="Decoding masks"):
        img_bytes = base64.b64decode(mask_data['data'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        masks_to_pack.append({
            'id': mask_data['id'],
            'width': mask_data['dimensions'][0],
            'height': mask_data['dimensions'][1],
            'image': mask_img
        })

    # Pack into atlas
    print("Packing texture atlas...")
    packer = TexturePacker()
    atlas_width, atlas_height = packer.pack(masks_to_pack)
    print(f"  Atlas size: {atlas_width}x{atlas_height}")

    # Create atlas image
    print("Rendering atlas...")
    atlas = np.zeros((atlas_height, atlas_width), dtype=np.uint8)

    uv_map = {}
    for rect in packer.rects:
        x, y = rect['x'], rect['y']
        w, h = rect['width'], rect['height']
        mask_img = rect['mask_data']['image']

        # Place in atlas
        atlas[y:y+h, x:x+w] = mask_img

        # Store UV mapping
        uv_map[rect['mask_id']] = {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }

    # Save atlas PNG with .atlas.png extension
    atlas_path = f"{output_prefix}.atlas.png"
    print(f"Saving atlas to {atlas_path}...")
    cv2.imwrite(atlas_path, atlas, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # Create metadata
    print("Creating metadata...")

    metadata = {
        'global_bbox': data['global_bbox'],
        'main_frame_id': main_frame_id,
        'atlas': {
            'width': atlas_width,
            'height': atlas_height,
            'file': Path(atlas_path).name
        },
        'uv_map': uv_map,
        'total_unique_part_ids' : data['total_unique_ids'],
        'frames': []
    }

    # Convert frames
    print(f"Converting {len(data['frames'])} frames...")
    for frame_data in tqdm(data['frames'], desc="Converting frames"):
        frame = {
            'index': frame_data['frame_index'],
            'black_mask_id': frame_data['black_lines_mask_id'],
            'parts': []
        }

        for part_data in frame_data['parts']:
            frame['parts'].append({
                'bbox': part_data['bbox'],
                'mask_id': part_data['mask_id'],
                'id': part_data['id']
            })

        metadata['frames'].append(frame)

    # Save metadata JSON
    metadata_path = f"{output_prefix}.meta.json"
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, separators=(',', ':'))

    # Print stats
    atlas_size = Path(atlas_path).stat().st_size
    metadata_size = Path(metadata_path).stat().st_size
    total_size = atlas_size + metadata_size

    print(f"\n{'='*50}")
    print(f"Atlas:    {atlas_size / 1024 / 1024:.1f} MB")
    print(f"Metadata: {metadata_size / 1024 / 1024:.1f} MB")
    print(f"Total:    {total_size / 1024 / 1024:.1f} MB")
    print(f"{'='*50}")

    return atlas_path, metadata_path


# ============================================================================
# IMAGE RESIZING AND SCALING
# ============================================================================

def resize_and_center_images(input_folder, output_folder, scale_factor, global_bbox):
    """
    Cut object by bbox, scale to Raccoon size, center in 1920x1080 white canvas

    Workflow:
    1. Cut the object by its bbox from the original image
    2. Scale the cut object to match Raccoon's bbox size
    3. Create a 1920x1080 white canvas
    4. Insert scaled object centered in the canvas
    5. Save to final_resize folder

    Args:
        input_folder: Path to input images
        output_folder: Path to save resized images
        scale_factor: Scale to apply to cut object
        global_bbox: Current creature's global bbox (where to cut)
    """
    print(f"\nCutting, scaling, and centering objects in 1920x1080 canvas...")
    print(f"  Scale factor: {scale_factor:.4f}")

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get Raccoon etalon data
    etalon_img_w, etalon_img_h = RACCOON_ETALON['image_size']  # 1920x1080
    etalon_bbox = RACCOON_ETALON['global_bbox']

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_folder.glob(f'*{ext}'))

    image_files = sorted(image_files, key=lambda x: x.name)

    for img_path in tqdm(image_files, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"  Warning: Could not read {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # Step 1: Cut the object by its bbox
        bbox_x = global_bbox['x']
        bbox_y = global_bbox['y']
        bbox_w = global_bbox['width']
        bbox_h = global_bbox['height']

        # Ensure bbox is within image bounds
        x1 = max(0, bbox_x)
        y1 = max(0, bbox_y)
        x2 = min(orig_w, bbox_x + bbox_w)
        y2 = min(orig_h, bbox_y + bbox_h)

        # Extract the bbox region
        cropped_obj = img[y1:y2, x1:x2].copy()

        # Step 2: Scale the cut object to match Raccoon's bbox size
        new_width = int(bbox_w * scale_factor)
        new_height = int(bbox_h * scale_factor)
        scaled_obj = cv2.resize(cropped_obj, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Step 3: Create 1920x1080 white canvas
        canvas = np.ones((etalon_img_h, etalon_img_w, 3), dtype=np.uint8) * 255

        # Step 4: Calculate position to center scaled object in Raccoon's bbox area
        # Center of Raccoon's bbox
        raccoon_center_x = etalon_bbox['x'] + etalon_bbox['width'] // 2
        raccoon_center_y = etalon_bbox['y'] + etalon_bbox['height'] // 2

        # Position to paste scaled object (centered)
        paste_x = raccoon_center_x - new_width // 2
        paste_y = raccoon_center_y - new_height // 2

        # Step 5: Paste scaled object on canvas
        # Calculate valid paste region
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(new_width, etalon_img_w - paste_x)
        src_y2 = min(new_height, etalon_img_h - paste_y)

        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = min(etalon_img_w, paste_x + new_width)
        dst_y2 = min(etalon_img_h, paste_y + new_height)

        # Copy the valid region
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_obj[src_y1:src_y2, src_x1:src_x2]

        # Step 6: Save to output
        output_path = output_folder / img_path.name
        cv2.imwrite(str(output_path), canvas)

    print(f"Processed {len(image_files)} images into 1920x1080 format")


# ============================================================================
# MAIN IMAGE GENERATION (FROM ATLAS)
# ============================================================================

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


def reconstruct_frame_from_atlas(atlas, metadata, frame_idx):
    """
    Reconstruct a single frame from atlas (black lines only)

    Args:
        atlas: Atlas image (grayscale)
        metadata: Metadata dictionary
        frame_idx: Frame index to reconstruct

    Returns:
        Reconstructed frame with black lines on transparent background (RGBA)
    """
    global_bbox = metadata['global_bbox']
    bbox_w = global_bbox['width']
    bbox_h = global_bbox['height']

    # Create transparent canvas for bbox region
    bbox_region = np.zeros((bbox_h, bbox_w, 4), dtype=np.uint8)

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
        bbox_region[black_pixels] = [0, 0, 0, 255]  # Black with full opacity

    return bbox_region


def create_main_from_atlas(atlas, metadata, frame_idx, folder_name, output_folder):
    """
    Create main image from atlas with white fill for colored parts and black lines.
    - Colored parts (from parts): white fill
    - Black lines: black
    - Everything else: transparent
    Crops to the bounding box of the object.

    Args:
        atlas: Atlas image (grayscale)
        metadata: Metadata dictionary
        frame_idx: Frame index to use for main image
        folder_name: Name for output file
        output_folder: Where to save the main image
    """
    print(f"\nCreating main image from frame {frame_idx}...")

    global_bbox = metadata['global_bbox']
    bbox_w = global_bbox['width']
    bbox_h = global_bbox['height']

    # Create transparent canvas for bbox region (BGRA)
    result = np.zeros((bbox_h, bbox_w, 4), dtype=np.uint8)

    # Get frame data
    frame_data = metadata['frames'][frame_idx]
    uv_map = metadata['uv_map']

    # Step 1: Draw all colored parts as white fill
    for part in frame_data['parts']:
        mask_id = str(part['mask_id'])
        if mask_id in uv_map:
            # Extract mask from atlas
            uv_coords = uv_map[mask_id]
            part_mask = extract_mask_from_atlas(atlas, uv_coords)

            # Get part's bbox position
            part_bbox = part['bbox']
            x, y, w, h = part_bbox[0], part_bbox[1], part_bbox[2], part_bbox[3]

            # Apply white fill where mask is white (pixels > 128)
            white_pixels = part_mask > 128
            # Place the mask at the correct position in the result
            result[y:y+h, x:x+w][white_pixels] = [255, 255, 255, 255]  # White with full opacity

    # Step 2: Draw black lines on top
    black_mask_id = str(frame_data['black_mask_id'])
    if black_mask_id in uv_map:
        uv_coords = uv_map[black_mask_id]
        black_mask = extract_mask_from_atlas(atlas, uv_coords)

        # Apply black lines: pixels > 128 in mask = black lines in output
        black_pixels = black_mask > 128
        result[black_pixels] = [0, 0, 0, 255]  # Black with full opacity

    # Find bounding box of non-transparent pixels
    alpha_channel = result[:, :, 3]
    points = np.where(alpha_channel > 0)

    if len(points[0]) == 0:
        print("  Warning: No content found in frame, using full frame")
        cropped = result
    else:
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()

        # Crop to content bbox
        cropped = result[y_min:y_max+1, x_min:x_max+1]

    crop_h, crop_w = cropped.shape[:2]
    print(f"  Main image size: {crop_w}x{crop_h} (white fill + black lines from atlas)")

    # Save
    output_path = output_folder / f"{folder_name}.main.png"
    cv2.imwrite(str(output_path), cropped)
    print(f"  Main image saved: {output_path}")

    return output_path




# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if len(sys.argv) != 3:
        print("Usage: python prepare_data.py path/to/folder frame_number")
        print("\nExample:")
        print("  python prepare_data.py colored/Bear 0")
        print("\nframe_number: Frame index to use for main image (e.g., 0, 1, 2...)")
        sys.exit(1)

    # Get folder path and frame number from arguments
    folder_path = Path(sys.argv[1])
    try:
        frame_number = int(sys.argv[2])
    except ValueError:
        print(f"Error: frame_number must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)


    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        sys.exit(1)

    # Get folder name for output files
    folder_name = folder_path.name

    # Define paths
    final_folder = folder_path / 'final'
    final_resize_folder = folder_path / 'final_resize'

    if not final_folder.exists():
        print(f"Error: '{final_folder}' folder not found!")
        sys.exit(1)

    # Get all image files from final folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(final_folder.glob(f'*{ext}'))

    # Sort files to ensure consistent ordering
    image_files = sorted(image_files, key=lambda x: x.name)

    if not image_files:
        print(f"No image files found in '{final_folder}' folder!")
        sys.exit(1)

    print(f"Found {len(image_files)} frames in {final_folder}")
    print("=" * 50)

    # Process all frames
    start_time = time.time()

    # ========================================
    # STEP 1: SMART RESIZE AND SAVE TO final_resize
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1: SMART RESIZE - Creating 1920x1080 images")
    print("=" * 70)

    # Get global bbox and calculate scale based on Raccoon etalon
    global_bbox = find_global_bounding_box(image_files)
    print(f"\nCalculating scale factor to fit into Raccoon bbox...")
    scale_factor = calculate_smart_scale_to_etalon(global_bbox)

    # Resize and center images in 1920x1080 canvas
    resize_and_center_images(final_folder, final_resize_folder, scale_factor, global_bbox)

    print(f"\n✓ Step 1 Complete: Resized images saved to {final_resize_folder}")

    # ========================================
    # STEP 2: CREATE ATLAS AND METADATA FROM final_resize
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: CREATE ATLAS AND METADATA from final_resize folder")
    print("=" * 70)

    resized_image_files = sorted(final_resize_folder.glob('*.png'), key=lambda x: x.name)

    if not resized_image_files:
        print("Error: No images found in final_resize folder!")
        sys.exit(1)

    print(f"Found {len(resized_image_files)} resized images in {final_resize_folder}")

    # Process resized images
    resized_data = process_frames_with_deduplication(resized_image_files)

    if resized_data:
        # Create atlas and metadata - output path without extension
        atlas_output_path = folder_path / folder_name
        atlas_path, metadata_path = create_atlas_and_metadata(resized_data, str(atlas_output_path), frame_number)

        print(f"\n✓ Step 2 Complete:")
        print(f"  Atlas:    {atlas_path}")
        print(f"  Metadata: {metadata_path}")
    else:
        print("Error: Failed to process resized images")
        sys.exit(1)

    # ========================================
    # STEP 3: CREATE MAIN IMAGE FROM ATLAS
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3: CREATE MAIN IMAGE from atlas")
    print("=" * 70)

    # Validate frame number
    if frame_number < 0 or frame_number >= len(resized_data['frames']):
        print(f"Error: Frame number {frame_number} is out of range (0-{len(resized_data['frames'])-1})")
        sys.exit(1)

    # Load atlas for main generation
    atlas = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
    if atlas is None:
        print(f"Error: Could not load atlas from {atlas_path}")
        sys.exit(1)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create main image from atlas
    main_path = create_main_from_atlas(
        atlas,
        metadata,
        frame_number,
        folder_name,
        folder_path
    )

    print(f"\n✓ Step 3 Complete:")
    print(f"  Main:    {main_path}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("=" * 50)


if __name__ == '__main__':
    main()
