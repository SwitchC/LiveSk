from AtlasManager import AtlasManager
import json
import cv2
import numpy as np
import colorsys
import math
import argparse
import sys

# --- ColorOptimizer Class ---
class ColorOptimizer:
    def __init__(self):
        self.palette = []
        golden_ratio_conjugate = 0.618033988749895
        h = 0.0
        for i in range(50):
            h += golden_ratio_conjugate
            h %= 1.0
            saturation = 0.6 + (i % 2) * 0.2
            value = 0.85 + (i % 2) * 0.1
            r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
            self.palette.append((int(b * 255), int(g * 255), int(r * 255)))

    def check_intersection(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        margin = 1
        return not (x1 + w1 + margin < x2 or x2 + w2 + margin < x1 or 
                    y1 + h1 + margin < y2 or y2 + h2 + margin < y1)

    def assign_colors(self, parts):
        """Assigns colors so neighbors have different colors."""
        assignments = {} 
        sorted_parts = sorted(parts, key=lambda p: (p.get('bbox', [0,0,0,0])[2] * p.get('bbox', [0,0,0,0])[3]), reverse=True)
        for part in sorted_parts:
            current_bbox = part.get('bbox')
            if not current_bbox or len(current_bbox) != 4: continue
            part_id = part.get('id')
            used_color_indices = set()
            for pid, assigned_idx in assignments.items():
                neighbor_part = next((p for p in parts if p['id'] == pid), None)
                if neighbor_part:
                    neighbor_bbox = neighbor_part.get('bbox')
                    if self.check_intersection(current_bbox, neighbor_bbox):
                        used_color_indices.add(assigned_idx)
            color_index = 0
            while color_index in used_color_indices:
                color_index += 1
                if color_index >= len(self.palette):
                    color_index = 0
                    break
            assignments[part_id] = color_index
        return {pid: self.palette[idx] for pid, idx in assignments.items()}

# --- LabelOptimizer Class ---
class LabelOptimizer:
    def __init__(self):
        self.labels = [] 

    def add_label(self, part_id, bbox, font_scale, thickness):
        """Registers a label for optimization."""
        x, y, w, h = bbox
        text = str(part_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        center_x = x + w / 2
        center_y = y + h / 2
        self.labels.append({
            'text': text, 'x': center_x, 'y': center_y, 'w': text_w, 'h': text_h,
            'orig_x': center_x, 'orig_y': center_y, 'radius': max(text_w, text_h) / 1.5
        })

    def optimize_positions(self, iterations=50):
        """
        Physical simulation:
        1. Repulsion between labels.
        2. Attraction to original center (so they don't wander too far).
        """
        width_padding = 5 # Extra padding between numbers
        for i in range(iterations):
            # Displacement force decreases with each iteration (cooling)
            alpha = 1.0 - (i / iterations)
            for j, l1 in enumerate(self.labels):
                force_x, force_y = 0, 0
                
                # 1. Repulsion from other labels
                for k, l2 in enumerate(self.labels):
                    if j == k: continue
                    dx = l1['x'] - l2['x']
                    dy = l1['y'] - l2['y']
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01 # +0.01 to avoid division by zero
                    min_dist = (l1['radius'] + l2['radius']) + width_padding
                    if dist < min_dist:
                        # Repulsion vector
                        push_force = (min_dist - dist) / dist
                        force_x += dx * push_force * 0.5
                        force_y += dy * push_force * 0.5
                
                # 2. Attraction to "home" (original bbox center)
                # Acts like a spring
                home_dx = l1['orig_x'] - l1['x']
                home_dy = l1['orig_y'] - l1['y']
                
                # Spring stiffness coefficient
                force_x += home_dx * 0.05
                force_y += home_dy * 0.05
                
                # Apply force
                l1['x'] += force_x * alpha
                l1['y'] += force_y * alpha

    def get_labels(self):
        return self.labels

# --- Main Logic ---

def draw_info_overlay(canvas, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = 1.0
    base_thickness = 2
    x_pos = 20
    y_pos = 40
    line_height = 35

    overlay = canvas.copy()
    total_h = len(lines) * line_height + 20
    cv2.rectangle(overlay, (0, 0), (600, total_h), (50, 50, 50), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    for line in lines:
        color = (255, 255, 255)
        if "Removed" in line: color = (150, 150, 255) # Reddish
        if "Added" in line: color = (150, 255, 150)   # Greenish
        cv2.putText(canvas, line, (x_pos, y_pos), font, base_scale, (0,0,0), base_thickness + 2)
        cv2.putText(canvas, line, (x_pos, y_pos), font, base_scale, color, base_thickness)
        y_pos += line_height

def generate_frame_image(manager, json_data, frame_index, overlay_lines):
    """Generates an image (numpy array) for a frame."""
    g_bbox = json_data.get('global_bbox', {'width': 1000, 'height': 1000})
    gw, gh = g_bbox['width'], g_bbox['height']
    canvas = np.ones((gh, gw, 3), dtype=np.uint8) * 255

    frames = json_data.get('frames', [])
    if frame_index >= len(frames): return canvas
    target_frame = frames[frame_index]
    parts = target_frame.get('parts', [])

    # 1. Colors and positions
    color_opt = ColorOptimizer()
    part_colors = color_opt.assign_colors(parts)
    label_opt = LabelOptimizer()
    part_font_params = {} 

    for part in parts:
        bbox = part.get('bbox')
        if not bbox or len(bbox) != 4: continue
        _, _, w, h = bbox
        min_dim = min(w, h)
        font_scale = max(0.4, min(0.9, min_dim / 35.0))
        thickness = max(1, int(font_scale * 2.5))
        part_font_params[part['id']] = (font_scale, thickness)
        label_opt.add_label(part['id'], bbox, font_scale, thickness)

    label_opt.optimize_positions(iterations=80)
    optimized_labels = {l['text']: l for l in label_opt.get_labels()}

    # 2. Drawing
    parts_sorted = sorted(parts, key=lambda p: (p.get('bbox', [0,0,0,0])[2] * p.get('bbox', [0,0,0,0])[3]), reverse=True)
    
    for part in parts_sorted:
        part_id = part.get('id')
        mask_id = part.get('mask_id')
        bbox = part.get('bbox')
        if not bbox or len(bbox) != 4: continue
        x, y, w, h = bbox
        color = part_colors.get(part_id, (128,128,128))
        part_mask = manager.get_mask(mask_id, as_numpy=True)
        if part_mask is not None:
            mask_resized = cv2.resize(part_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_rect = np.full((h, w, 3), color, dtype=np.uint8)
            y1, y2 = y, y + h
            x1, x2 = x, x + w
            if x1 < gw and y1 < gh:
                roi = canvas[y1:y2, x1:x2]
                mask_bool = mask_resized > 10 
                roi[mask_bool] = color_rect[mask_bool]
                canvas[y1:y2, x1:x2] = roi

    black_mask_id = target_frame.get('black_mask_id')
    if black_mask_id:
        black_lines_img = manager.get_mask(black_mask_id, as_numpy=True)
        if black_lines_img is not None:
            black_lines_resized = cv2.resize(black_lines_img, (gw, gh))
            canvas[black_lines_resized > 200] = [0, 0, 0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for part in parts_sorted:
        part_id = part.get('id')
        str_id = str(part_id)
        if str_id in optimized_labels:
            label_data = optimized_labels[str_id]
            font_scale, thickness = part_font_params[part_id]
            color = part_colors.get(part_id, (0,0,0))
            text_x = int(label_data['x'] - label_data['w'] / 2)
            text_y = int(label_data['y'] + label_data['h'] / 2)
            cv2.putText(canvas, str_id, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 4)
            cv2.putText(canvas, str_id, (text_x, text_y), font, font_scale, color, thickness)

    draw_info_overlay(canvas, overlay_lines)
    return canvas

def create_combined_debug_image(json_path, atlas_dir=None, output_file="debug_grid.png"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return

    if atlas_dir: manager = AtlasManager(data, atlas_dir=atlas_dir)
    else: manager = AtlasManager(data)
    if manager.atlas_image is None: return

    frames = data.get('frames', [])
    if not frames: return

    g_bbox = data.get('global_bbox', {'width': 1000, 'height': 1000})
    gw, gh = g_bbox['width'], g_bbox['height']

    # List to store all generated images
    images_list = []
    
    # 1. Frame 0
    frame0_parts = frames[0].get('parts', [])
    base_ids = set(p['id'] for p in frame0_parts)
    
    print("Generating Frame 0...")
    img0 = generate_frame_image(manager, data, 0, ["Frame 0", "Base Layout"])
    images_list.append(img0)

    # Initialize previous state (empty difference)
    prev_removed = set()
    prev_added = set()

    # 2. Analyze changes
    for i in range(1, len(frames)):
        current_parts = frames[i].get('parts', [])
        current_ids = set(p['id'] for p in current_parts)
        
        # Calculate current difference from Frame 0
        removed = base_ids - current_ids 
        added = current_ids - base_ids   

        # Logic: 
        # 1. There must be a difference from Frame 0 (removed or added is not empty).
        # 2. This difference must NOT be identical to the previous frame's difference.
        has_changes_vs_base = bool(removed or added)
        is_same_as_prev = (removed == prev_removed) and (added == prev_added)

        if has_changes_vs_base and not is_same_as_prev:
            print(f"Detecting new changes in Frame {i}...")
            lines = [f"Frame {i}"]
            if removed: lines.append(f"Removed: {sorted(list(removed))}")
            if added: lines.append(f"Added: {sorted(list(added))}")
            img = generate_frame_image(manager, data, i, lines)
            images_list.append(img)
        
        # Update previous state for next iteration
        prev_removed = removed
        prev_added = added

    # --- 3. Grid Layout Concatenation ---
    if not images_list:
        print("No images generated.")
        return

    COLUMNS = 3 # Number of columns in the grid
    total_images = len(images_list)
    # Calculate required rows (ceiling division)
    ROWS = (total_images + COLUMNS - 1) // COLUMNS

    grid_rows = []
    current_row_images = []

    for i, img in enumerate(images_list):
        current_row_images.append(img)

        # If the row is full OR this is the very last image
        if len(current_row_images) == COLUMNS or i == total_images - 1:
            
            # If it's the last row and it's not full, fill with empty squares
            while len(current_row_images) < COLUMNS:
                # Empty white square of the same size
                blank_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255
                current_row_images.append(blank_img)
            
            # Concatenate row horizontally
            row_concat = cv2.hconcat(current_row_images)
            grid_rows.append(row_concat)
            current_row_images = []

    # Concatenate rows vertically
    if grid_rows:
        print(f"Stacking into grid: {ROWS} rows x {COLUMNS} columns...")
        final_image = cv2.vconcat(grid_rows)
        cv2.imwrite(output_file, final_image)
        print(f"Done! Saved grid view to: {output_file}")


# --- RUN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a debug grid for atlas animation changes.")
    parser.add_argument("json_path", help="Path to the meta.json file")
    parser.add_argument("--atlas_dir", default=None, help="Directory containing the atlas image (default: same as script)")
    parser.add_argument("--output", default="parts_id.png", help="Output filename (default: parts_id.png)")
    
    args = parser.parse_args()
    
    # If atlas_dir is not provided, use the current directory
    atlas_directory = args.atlas_dir if args.atlas_dir else "."
    
    create_combined_debug_image(args.json_path, atlas_directory, args.output)