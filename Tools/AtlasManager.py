import os
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class AtlasManager:
    def __init__(self, atlas_data, atlas_dir=None):
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
            self.atlas_image.load() 
        except Exception as e:
            print(f"Critical error: {e}")
            self.atlas_image = None

    def get_mask(self, mask_id, as_numpy=False):
        if self.atlas_image is None: return None
        str_id = str(mask_id)
        if str_id not in self.uv_map: return None
        coords = self.uv_map[str_id]
        crop_box = (coords['x'], coords['y'], coords['x'] + coords['width'], coords['y'] + coords['height'])
        cropped_mask = self.atlas_image.crop(crop_box)
        if as_numpy:
            arr = np.array(cropped_mask)
            if cropped_mask.mode == 'RGBA': arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            elif cropped_mask.mode == 'RGB': arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr
        return cropped_mask.convert('L')
