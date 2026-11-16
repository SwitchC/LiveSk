import cv2
import numpy as np
from pathlib import Path
import re


def natural_sort_key(filename):
    """Sort filenames naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(filename))]


def load_frames_from_folder(folder_path):
    """Load all image frames from folder, sorted naturally"""
    folder = Path(folder_path)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    frame_files = [f for f in folder.iterdir()
                   if f.suffix.lower() in image_extensions]
    frame_files = sorted(frame_files, key=natural_sort_key)

    print(f"Found {len(frame_files)} frames")

    frames = []
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            frames.append(frame)

    return frames


def generate_100_colors():
    """Generate 100 distinct colors"""
    colors = []
    np.random.seed(42)

    # Generate vibrant, distinct colors
    for i in range(100):
        # Use HSV color space for better distribution
        hue = (i * 137.5) % 360  # Golden angle for good distribution
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 4) * 0.1

        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)

        colors.append((b, g, r))  # OpenCV uses BGR

    return colors


def detect_parts_in_frame(frame, min_area=200):
    """
    Detect all white parts in a single frame
    Returns list of parts with their properties
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to get white regions
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    parts = []

    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h, area = stats[i]

        # Skip if this is the entire frame border
        if h == frame.shape[0] or w == frame.shape[1]:
            continue

        # Skip small parts
        if area < min_area:
            continue

        # Create mask for this part
        part_mask = (labels == i).astype(np.uint8) * 255

        parts.append({
            'label': i,
            'centroid': centroids[i],
            'bbox': (x, y, w, h),
            'area': area,
            'mask': part_mask
        })

    return parts


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def match_parts(prev_parts, curr_parts, max_distance=50, min_iou=0.3):
    """
    Match parts between consecutive frames
    Returns: dict mapping curr_part_idx -> prev_part_idx (or None for new parts)
    """
    matches = {}

    if not prev_parts:
        # First frame - no matches
        return {i: None for i in range(len(curr_parts))}

    # Calculate matching scores for all pairs
    match_scores = []

    for curr_idx, curr_part in enumerate(curr_parts):
        for prev_idx, prev_part in enumerate(prev_parts):
            # Distance between centroids
            centroid_dist = np.linalg.norm(
                np.array(curr_part['centroid']) - np.array(prev_part['centroid'])
            )

            # Area similarity
            area_ratio = min(curr_part['area'], prev_part['area']) / max(curr_part['area'], prev_part['area'])

            # IoU (Intersection over Union) - best indicator
            iou = calculate_iou(curr_part['mask'], prev_part['mask'])

            # Combined score (IoU is most important)
            if centroid_dist < max_distance and iou > min_iou:
                score = iou * 0.7 + area_ratio * 0.2 + (1 - centroid_dist / max_distance) * 0.1
                match_scores.append((score, curr_idx, prev_idx))

    # Sort by score (highest first)
    match_scores.sort(reverse=True)

    # Assign matches (greedy approach - best matches first)
    used_prev = set()
    used_curr = set()

    for score, curr_idx, prev_idx in match_scores:
        if curr_idx not in used_curr and prev_idx not in used_prev:
            matches[curr_idx] = prev_idx
            used_curr.add(curr_idx)
            used_prev.add(prev_idx)

    # Remaining current parts are new
    for curr_idx in range(len(curr_parts)):
        if curr_idx not in matches:
            matches[curr_idx] = None

    return matches


def process_all_frames_with_tracking(frames, output_folder):
    """
    Process all frames with tracking - same parts get same colors
    Uses first frame as reference for all subsequent frames
    Unmatched parts stay WHITE
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Generate colors
    colors = generate_100_colors()
    print(f"Generated {len(colors)} colors")

    # Track part IDs and their assigned colors (only for first frame parts)
    part_id_to_color = {}

    # First frame - reference frame
    first_frame_parts = None
    first_frame_part_ids = None

    print("\nProcessing frames with tracking...")
    print("Using first frame as reference - unmatched parts will be WHITE\n")

    for frame_idx, frame in enumerate(frames):
        print(f"Processing frame {frame_idx + 1}/{len(frames)}", end='\r')

        # Detect parts in current frame
        curr_parts = detect_parts_in_frame(frame)

        if frame_idx == 0:
            # First frame - assign colors to all parts
            first_frame_parts = curr_parts
            first_frame_part_ids = list(range(len(curr_parts)))

            # Assign colors to all first frame parts
            for part_id in first_frame_part_ids:
                part_id_to_color[part_id] = colors[part_id % len(colors)]

        # Create result image (WHITE background)
        result = np.ones_like(frame) * 255

        # First, copy the original black borders/lines from the sketch
        # (any pixel that's not white in the original)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_lines = gray < 200  # Everything that's not white
        result[black_lines] = frame[black_lines]

        if frame_idx == 0:
            # First frame - color all parts
            for part_idx, part in enumerate(curr_parts):
                part_id = first_frame_part_ids[part_idx]
                color = part_id_to_color[part_id]

                # Color the part (preserving original contours)
                result[part['mask'] > 0] = color
        else:
            # Subsequent frames - match with first frame
            matches = match_parts(first_frame_parts, curr_parts)

            for curr_idx, curr_part in enumerate(curr_parts):
                first_frame_idx = matches[curr_idx]

                if first_frame_idx is not None:
                    # Matched with first frame part - use same color
                    part_id = first_frame_part_ids[first_frame_idx]
                    color = part_id_to_color[part_id]
                else:
                    # No match - use WHITE
                    color = (255, 255, 255)

                # Color the part (preserving original contours)
                result[curr_part['mask'] > 0] = color

        # Re-apply black borders on top to ensure they're visible
        result[black_lines] = frame[black_lines]

        # Save frame
        output_path = output_folder / f"colored_{frame_idx:04d}.png"
        cv2.imwrite(str(output_path), result)

    print(f"\nComplete! Processed {len(frames)} frames")
    print(f"First frame had {len(first_frame_part_ids)} parts with assigned colors")
    print(f"Parts not matching first frame will appear WHITE")


def main():
    """Main function"""
    # === CONFIGURATION ===
    input_folder = "frames"
    output_folder = "colored_frames"

    print("=== White Parts Detection and Tracking ===\n")

    # Load frames
    print(f"Loading frames from: {input_folder}")
    frames = load_frames_from_folder(input_folder)

    if not frames:
        print("Error: No frames found!")
        return

    print(f"Loaded {len(frames)} frames\n")

    process_all_frames_with_tracking(frames, output_folder)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) >= 3 else "colored"

        print("=== White Parts Detection and Tracking ===\n")
        print(f"Loading frames from: {input_folder}")
        frames = load_frames_from_folder(input_folder)

        if not frames:
            print("Error: No frames found!")
        else:
            print(f"Loaded {len(frames)} frames\n")
            process_all_frames_with_tracking(frames, output_folder)
    else:
        main()