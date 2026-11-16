# Constants for image resizing and processing
#
# ETALON (Reference) - Raccoon
# All other creatures will be scaled to match Raccoon's proportions

# Raccoon reference data (etalon)
RACCOON_ETALON = {
    'image_size': (1080, 1920),  # width x height
    'global_bbox': {
        'x': 54,
        'y': 368,
        'width': 979,
        'height': 1185
    }
}

# This is the etalon that all other folders will be scaled to match
# The script will calculate scale factors so that each creature's bbox
# matches the same proportions as Raccoon's bbox relative to image size

# Image processing thresholds
BLACK_THRESHOLD = 30  # Threshold for detecting black lines
WHITE_THRESHOLD = 240  # Threshold for detecting white background
BBOX_PADDING = 10  # Padding around bounding boxes

# Atlas settings
MAX_ATLAS_WIDTH = 8192  # Maximum atlas texture width

# Output image sizes
OUTPUT_IMAGE_SIZE = (1080, 1920)  # Width x Height for output images (final_resize)
