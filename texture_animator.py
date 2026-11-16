#!/usr/bin/env python3
"""
Texture Animator - Optimized version
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from AtlasManager import AtlasManager

class TextureAnimator:
    """Optimized animator class"""

    def __init__(self, data_path: str,transform_path: str, first_frame_path: str, output_dir: str):
        """Initialize the animator"""
        self.data_path = Path(data_path)
        self.transform_path = Path(transform_path)
        self.first_frame_path = Path(first_frame_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading data from '{self.data_path}'...")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # Load transform
        print(f"Loading transform from '{self.transform_path}'...")
        with open(self.transform_path, 'r') as f:
            self.transform = json.load(f)
        
        # Initialize atlas
        self.atlas = AtlasManager(self.data, atlas_dir=self.data_path.parent)

        # Load first frame with drawing
        raw_image = cv2.imread(str(self.first_frame_path), cv2.IMREAD_UNCHANGED)
        
        if raw_image is None:
            raise ValueError(f"Could not load image from {self.first_frame_path}")

        # 2. Check for alpha channel (4th channel)
        if raw_image.shape[2] == 4:
            print("Detected RGBA image. Converting to RGB with WHITE background...")
            
            # Split channels
            b, g, r, a = cv2.split(raw_image)
            foreground = cv2.merge((b, g, r))
            
            # Create white background of the same size
            white_background = np.ones_like(foreground, dtype=np.uint8) * 255
            
            # --- Alpha blending (best method) ---
            # This removes "dirty" pixels on the edges.
            # Formula: pixel = (fg * alpha) + (bg * (1 - alpha))
            
            alpha_factor = a.astype(float) / 255.0
            alpha_factor = np.stack([alpha_factor] * 3, axis=2) # Робимо 3 канали для множення
            
            final_image = (foreground.astype(float) * alpha_factor) + \
                          (white_background.astype(float) * (1.0 - alpha_factor))
            
            # Save as final 3-channel BGR image
            self.input_image = final_image.astype(np.uint8)
            
        else:
            # If channels are already 3 - just use as is
            self.input_image = raw_image
        if self.input_image is None:
            raise ValueError(f"Could not load image from {self.first_frame_path}")

        # Global bbox info
        self.global_bbox = self.data['global_bbox']

        # Resize input image to match global bbox if needed
        self.gw = self.global_bbox['width']
        self.gh = self.global_bbox['height']

        if self.input_image.shape[:2] != (self.gh, self.gw):
            print(f"Adapting input image to global size: ({self.gh}, {self.gw})")
            
            # 1. Get the mask and BBox where the image needs to be inserted
            # Important: get_mask must return a numpy array (as_numpy=True)
            black_lines_mask = self.atlas.get_mask(self.data['frames'][0]['black_mask_id'], as_numpy=True)
            
            x, y, w, h = self.get_bbox(black_lines_mask)
            
            # 2. Create an empty black canvas of the required size
            if len(self.input_image.shape) == 3:
                # If the image is colored (H, W, 3)
                channels = self.input_image.shape[2]
                new_canvas = np.zeros((self.gh, self.gw, channels), dtype=self.input_image.dtype)
            else:
                # If the image is grayscale (H, W)
                new_canvas = np.zeros((self.gh, self.gw), dtype=self.input_image.dtype)

            # 3. Resize input image to match BBox size
            # cv2.resize accepts dimensions in (width, height) format
            resized_input = cv2.resize(self.input_image, (w, h), interpolation=cv2.INTER_AREA)

            # 4. Insert the image into the bbox on the new canvas
            new_canvas[y : y + h, x : x + w] = resized_input

            self.input_image = new_canvas

        # Store extracted textures for each part ID
        self.part_textures = {}

        # Pre-allocate reusable arrays
        self.canvas_template = np.ones((self.gh, self.gw, 3), dtype=np.uint8) * 255

        self.fully_filled_parts = set()

        print(f"Global bbox: w={self.gw}, h={self.gh}")
        print(f"Found {len(self.data['frames'])} frames")
        print(f"Mask library contains {len(self.data['uv_map'])} unique masks")
    
    def get_bbox(self, mask):
        """
        Finds the bounding box (bbox) of a white object on a black background.
        Returns a tuple: (x, y, w, h)
        """
        # 1. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            return None  # If no objects are found
    
        # 2. If there is a single object, take the first contour.
        # If the object is fragmented but a unified bbox is needed, merge them.
    
        # BBox of the largest fragment
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return x, y, w, h
    def extract_textures_from_first_frame(self):
        """Extract textures from first frame using masks"""
        print("\nExtracting textures from first frame...")

        if not self.data['frames']:
            print("No frames found in data")
            return
# --- ЗБЕРЕЖЕННЯ ПРЯМО НА ДИСК C:\ ---
        output_dir = r"C:\Extracted_Textures"
        
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"Папка для збереження: {output_dir}")
        except PermissionError:
            print(f"ПОМИЛКА ДОСТУПУ: Windows не дозволяє створити папку на диску C:\\.")
            print("Рішення: Запустіть ваш редактор (PyCharm/VS Code/Термінал) від імені Адміністратора!")
            return
        except Exception as e:
            print(f"Інша помилка створення папки: {e}")
            return
            
        # Збереження оригінального малюнка для перевірки
        if hasattr(self, 'input_image') and self.input_image is not None:
            cv2.imwrite(os.path.join(output_dir, "0_original_input.png"), self.input_image)
        # --------------------------------------------

        if hasattr(self, 'input_image') and self.input_image is not None:
            cv2.imwrite(os.path.join(output_dir, "0_original_input.png"), self.input_image)
            print("SAVED: Original input image saved to desktop for verification.")
        else:
            print("Warning: self.input_image is not found or empty")

        first_frame = self.data['frames'][0]
  
        print(f"First frame has {len(first_frame['parts'])} parts")

        for i in range(len(first_frame['parts'])):

            part = first_frame['parts'][i]

            if part['id'] in self.data['non_drawable_part_ids']:
                continue

            bbox = part['bbox']
            mask_id = part['mask_id']

            part_mask = self.atlas.get_mask(mask_id, as_numpy = True)

            if part_mask is None:
                continue

            x, y, w, h = bbox
            mask_h, mask_w = part_mask.shape[:2]

            y_end = min(y + mask_h, self.gh)
            x_end = min(x + mask_w, self.gw)
            y_start = max(0, y)
            x_start = max(0, x)

            valid_h = y_end - y_start
            valid_w = x_end - x_start

            if valid_h <= 0 or valid_w <= 0:
                continue

            mask_region = part_mask[:(y_end - y), :(x_end - x)]
            input_region = self.input_image[y_start:y_end, x_start:x_end]

            # Create white texture
            extracted_texture = np.full_like(input_region, 255)

            # Single boolean mask calculation
            mask_bool = mask_region > 128
            gray_input = cv2.cvtColor(input_region, cv2.COLOR_BGR2GRAY)
            valid_pixels = mask_bool & (gray_input >= 30)

            # Copy all channels at once using numpy indexing
            extracted_texture[valid_pixels] = input_region[valid_pixels]

            # Check if fully filled (optimized)
            gray = cv2.cvtColor(extracted_texture, cv2.COLOR_BGR2GRAY)
            if np.sum((gray > 252) & mask_bool) < 1000:
                self.fully_filled_parts.add(i)

            self.part_textures[first_frame['parts'][i]['id']] = {
                'texture': self.resize_and_center(extracted_texture),
                'mask': mask_region,
                'original_bbox': bbox,
                'original_mask_id': mask_id
            }

        print(f"Extracted textures for {len(self.part_textures)} parts")
    def resize_sizes(self, w, h, target_size=500):
            new_h, new_w = h, w
            if h > target_size or w > target_size:
                scale = min(target_size / h, target_size / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
            return (new_w, new_h)
    def resize_and_center(self, image, target_size = 600):
        """
        Scales the image and places it EXACTLY IN THE CENTER 
        of a target_size x target_size canvas. Supports RGB, RGBA, and Grayscale.
        """
        h, w = image.shape[:2] # Get only height and width

        # --- STEP 1: Scaling ---
        if h > target_size or w > target_size:
            scale = min(target_size / h, target_size / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # --- STEP 2: Create Canvas (FIXED) ---
        # Check if image has color channels (RGB/RGBA)
        if len(image.shape) == 3:
            # image.shape = (h, w, channels)
            channels = image.shape[2]
            # Create 3D canvas: (target_size, target_size, channels)
            canvas = np.full((target_size, target_size, channels), 255, dtype=image.dtype)
        else:
            # Grayscale, shape = (h, w)
            # Create 2D canvas: (target_size, target_size)
            canvas = np.full((target_size, target_size), 255, dtype=image.dtype)

        # --- STEP 3: Centering ---
        pad_top = (target_size - h) // 2
        pad_left = (target_size - w) // 2

        # Insert the image
        # NumPy automatically fills all channels if dimensions match
        canvas[pad_top : pad_top + h, pad_left : pad_left + w] = image

        return canvas
    def apply_affine_transform(self, image_np, trained_matrix_np):
        """
        Applies the affine matrix using OpenCV.
        Automatically converts the matrix from PyTorch format ([-1, 1]) to OpenCV format (pixels).
        """
        # --- INPUT FORMAT FIX ---
        # Check if the image has (3, H, W) format - like in PyTorch
        if image_np.ndim == 3 and image_np.shape[0] in [3, 4] and image_np.shape[2] > 4:
            # If the first dimension is 3 or 4, and the last is large (width) -> it's (C, H, W)
            # Transpose to (H, W, C) for OpenCV
            image_np = np.transpose(image_np, (1, 2, 0))
        
        h, w = image_np.shape[:2]

        # 1. Convert matrix from PyTorch coordinates to OpenCV pixels
        # PyTorch matrix (2x3) works with coordinates from -1.0 to 1.0.
        # OpenCV warpAffine expects coordinates in pixels (0..W, 0..H)

        # Extend the matrix to 3x3 (add row [0, 0, 1])
        stn_matrix_3x3 = np.vstack([trained_matrix_np, [0, 0, 1]])

        # Transformation matrix: Pixels -> Normalized (-1..1)
        # x_norm = (x_pix / (w/2)) - 1
        pixel_to_norm = np.array([
            [2/w, 0,   -1],
            [0,   2/h, -1],
            [0,   0,    1]
        ])

        # Transformation matrix: Normalized -> Pixels
        # x_pix = (x_norm + 1) * (w/2)  
        norm_to_pixel = np.array([
            [w/2, 0,   w/2],
            [0,   h/2, h/2],
            [0,   0,   1]
        ])

        # Final matrix for OpenCV: M = Norm2Pixel * STN_Matrix * Pixel2Norm
        # Multiplication order is important (right-to-left according to application logic)
        opencv_matrix_3x3 = norm_to_pixel @ stn_matrix_3x3 @ pixel_to_norm
        
        # Take only the first 2 rows for cv2.warpAffine
        opencv_affine_matrix = opencv_matrix_3x3[:2, :]

        # 2. Apply transformation
        # STN uses "Inverse Mapping" (Sampling), so we add the WARP_INVERSE_MAP flag.
        # borderValue=(0,0,0) makes the background black (like padding_mode='zeros' in PyTorch)
        transformed_image = cv2.warpAffine(
            image_np, 
            opencv_affine_matrix, 
            (w, h), 
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue = (0, 0, 0)
        )

        return transformed_image

    def crop_center(self, image, h, w):
        """
        Crops the central part of the image (H, W).
        """
        img_h, img_w = image.shape[:2]

        if h > img_h or w > img_w:
            print(f"Warning: Crop ({h}x{w}) > Image ({img_h}x{img_w}). Returning original.")
            return image

        start_y = (img_h - h) // 2
        start_x = (img_w - w) // 2

        return image[start_y : start_y + h, start_x : start_x + w]

    def apply_textures_to_frame(self, frame_data, transform, is_first_frame=False):
        """Apply extracted textures to a frame using mask shapes"""
        reconstructed = self.canvas_template.copy()

        for i in range(len(frame_data['parts'])):
            part = frame_data['parts'][i]

            if part['id'] not in self.part_textures:
                continue
            
            if part['id'] in self.data['non_drawable_part_ids']:
                continue

            texture_data = self.part_textures[part['id']]
            original_texture = texture_data['texture']
            original_mask = texture_data['mask']
            
            adapted_texture = None
            current_mask_id = part['mask_id']

            part_t = None
            
            for t in transform[frame_data['index'] - 1]:
                if t['id'] == part['id']:
                    part_t = t
                    break
            
            if part_t is None:
                continue
            
            transformed_original_texture = self.apply_affine_transform(original_texture,np.array(part_t["matrix"]))
            current_mask = self.atlas.get_mask(current_mask_id, as_numpy = True)
            mask_h, mask_w = current_mask.shape[:2]
            sizes = self.resize_sizes(mask_w,mask_h,600)
            adapted_texture = self.crop_center(transformed_original_texture , sizes[1], sizes[0])

            
            current_bbox = part['bbox']
            current_mask_id = part['mask_id']

            current_mask = self.atlas.get_mask(current_mask_id, as_numpy = True)
            if current_mask is None:
                continue

            x, y, w, h = current_bbox
            
            # --- Check dimensions validity ---
            if w <= 0 or h <= 0:
                continue
                
            orig_h, orig_w = original_texture.shape[:2]
            if orig_h <= 0 or orig_w <= 0:
                continue
            # --- 1. Scale texture and mask to bbox size (w, h) ---

            # Resize texture to the insertion area size
            resized_texture = cv2.resize(adapted_texture, (w,h))
            # Resize mask to the same size.
            # INTER_NEAREST preserves sharp edges (0 or >0), which is important for masks
            resized_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- 2. Safe coordinate definition (to avoid going out of bounds) ---
            h_bg, w_bg = reconstructed.shape[:2]
            
            # Calculate actual intersection coordinates between bbox and canvas
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # If the area is invalid, skip
            if x1 >= x2 or y1 >= y2:
                continue

            # --- 3. Crop corresponding parts ---
            # We need to crop parts if bbox partially goes out of bounds
            # (e.g., y < 0), so we use offsets relative to x and y
            tex_crop = resized_texture[y1-y : y2-y, x1-x : x2-x]
            mask_crop = resized_mask[y1-y : y2-y, x1-x : x2-x]
            roi = reconstructed[y1:y2, x1:x2] # Region of Interest on the background

            # --- 4. Overlay with mask ---
            
            # Create condition: pixels visible where mask > 0
            is_visible = mask_crop > 0

            # If roi has 3 channels (color), and mask has 2 channels (black/white), add an axis
            if len(roi.shape) == 3 and len(is_visible.shape) == 2:
                is_visible = is_visible[:, :, np.newaxis]

            # np.where(condition, if_true, if_false)
            # If mask > 0, take pixel from texture (tex_crop), otherwise keep background (roi)
            reconstructed[y1:y2, x1:x2] = np.where(is_visible, tex_crop, roi)

        # Apply black lines
        black_mask_id = frame_data['black_mask_id']
        black_mask = self.atlas.get_mask(black_mask_id,as_numpy=True)

        if black_mask is not None and black_mask.shape[:2] == (self.gh, self.gw):
            black_pixels = black_mask > 200
            reconstructed[black_pixels] = 0

        return reconstructed

    def animate(self):
        """Optimized animation process"""
        print(f"\nStarting animation process...")
        print(f"Output directory: {self.output_dir}")

        # Extract textures from first frame
        self.extract_textures_from_first_frame()

        if not self.part_textures:
            print("Error: No textures extracted from first frame")
            return

        print(f"\nProcessing {len(self.data['frames'])} frames...")

        frames = self.data['frames']
        successful_frames = 0

        # Process frames with progress indicator
        for i, frame_data in tqdm(enumerate(frames)):

            frame_idx = frame_data['index']

            try:

                output_frame = self.apply_textures_to_frame(frame_data, self.transform)

                # Save output frame
                output_path = self.output_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(output_path), output_frame)
                successful_frames += 1

            except Exception as e:
                print(f"  Error processing frame {frame_idx}: {e}")

        print(f"\nAnimation complete!")
        print(f"Successfully processed: {successful_frames}/{len(self.data['frames'])} frames")

        # Create a video from frames
        self.create_video()

    def create_video(self, fps: int = 30):
        """Create video from output frames"""
        print("\nCreating video from frames...")

        frame_files = sorted(self.output_dir.glob("frame_*.png"))
        if not frame_files:
            print("No frames found to create video")
            return

        # Read first frame to get dimensions
        first = cv2.imread(str(frame_files[0]))
        if first is None:
            print("Could not read first frame")
            return

        height, width = first.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = self.output_dir / "animation.mp4"
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Write frames
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"Video saved: {video_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Texture Animator - Optimized Version")
    parser.add_argument("data_json", help="Path to meta.json file")
    parser.add_argument("transform_json", help="Path to transform.json file")
    parser.add_argument("first_frame", help="Path to first frame image with drawing")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video (default: 30)")


    args = parser.parse_args()

    # Validate inputs
    if not Path(args.data_json).exists():
        print(f"Error: Data file not found: {args.data_json}")
        sys.exit(1)
    
    if not Path(args.transform_json).exists():
        print(f"Error: Transform file not found: {args.transform_json}")
        sys.exit(1)

    if not Path(args.first_frame).exists():
        print(f"Error: First frame image not found: {args.first_frame}")
        sys.exit(1)

    try:
        # Create animator and run
        animator = TextureAnimator(
            data_path=args.data_json,
            transform_path=args.transform_json,
            first_frame_path=args.first_frame,
            output_dir=args.output
        )

        animator.animate()

    except Exception as e:
        print(f"Error during animation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()