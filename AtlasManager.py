import os
from PIL import Image
import numpy as np
import cv2

# Disable the limit for maximum image size
Image.MAX_IMAGE_PIXELS = None

class AtlasManager:
    def __init__(self, atlas_data, atlas_dir=None):
        """
        Initialization of the atlas manager.
        """
        self.uv_map = atlas_data.get('uv_map', {})
        
        if "atlas" in atlas_data and "file" in atlas_data["atlas"]:
             self.file_name = atlas_data["atlas"]["file"]
        else:
             self.file_name = atlas_data.get('file', 'atlas.png')

        self.parts = atlas_data.get('parts', [])
        
        if atlas_dir:
            image_path = os.path.join(atlas_dir, self.file_name)
        else:
            image_path = self.file_name

        try:
            self.atlas_image = Image.open(image_path)
            # Force loading to verify data integrity
            self.atlas_image.load() 
            print(f"Atlas loaded: {self.atlas_image.size}, mode: {self.atlas_image.mode}")
        except Exception as e:
            print(f"Critical error while opening atlas: {e}")
            self.atlas_image = None

    def get_mask(self, mask_id, as_numpy=False):
        """ 
        Retrieves a mask by its ID.
        If as_numpy=True, returns a single-channel array (Grayscale) of shape (H, W).
        """
        if self.atlas_image is None:
            return None

        str_id = str(mask_id)
        if str_id not in self.uv_map:
        # print(f"Error: ID {mask_id} not found in uv_map.")
            return None
        
        coords = self.uv_map[str_id]
        x, y = coords['x'], coords['y']
        w, h = coords['width'], coords['height']
        crop_box = (x, y, x + w, y + h)
        
        # 1. Crop the piece using PIL
        cropped_mask = self.atlas_image.crop(crop_box)
        
        if as_numpy:
            arr = np.array(cropped_mask)
            
            # --- GRAYSCALE CONVERSION LOGIC ---
            # PIL returns arrays in RGB or RGBA format.
            # OpenCV requires a single channel for working with masks.
            
            if cropped_mask.mode == 'RGBA':
               # Important: Use RGB2GRAY because the data came from PIL (RGB order),
                # rather than OpenCV (BGR order).
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            
            elif cropped_mask.mode == 'RGB':
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            
            # If mode == 'L' (Luminance/Grayscale), arr is already 2D (H, W)
            # and no conversion is needed.
                
            return arr
            
        # If as_numpy=False, return PIL object - convert to 'L' (8-bit pixels, black and white)
        return cropped_mask.convert('L')

    def get_part_image(self, part_index, as_numpy=False):
        """ Retrieves a mask by its index in the parts list """
        if part_index < 0 or part_index >= len(self.parts):
            return None

        part = self.parts[part_index]
        mask_id = part.get('mask_id')
        
        if mask_id is None:
            return None
            
        return self.get_mask(mask_id, as_numpy)