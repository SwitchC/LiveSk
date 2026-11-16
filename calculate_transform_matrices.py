import argparse
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AtlasManager import AtlasManager
# --- 1. OneSidedLoss ---
class OneSidedLoss(nn.Module):
    def __init__(self):
        super(OneSidedLoss, self).__init__()

    def forward(self, source, target):
        diff = target - source
        loss_component = F.relu(diff)
        loss = torch.mean(loss_component ** 2)
        return loss

class AffineRegistrationModel(nn.Module):
    # initial_theta for Warm Start
    def __init__(self, initial_theta=None):
        super(AffineRegistrationModel, self).__init__()
        
        if initial_theta is not None:
            self.theta = nn.Parameter(torch.tensor(initial_theta, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ], dtype=torch.float32))

    def forward(self, x):
        theta_batch = self.theta.unsqueeze(0).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(theta_batch, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False, mode='bilinear')
        return x_transformed

# --- 2. Helpers ---

def prepare_single_channel_tensor(image_np):
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    return img_tensor.unsqueeze(0)

def resize_and_center(image, target_size = 600):
    if image is None or image.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=image.dtype)

    if h > target_size or w > target_size:
        scale = min(target_size / h, target_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w > 0 and new_h > 0:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

    canvas = np.zeros((target_size, target_size), dtype=image.dtype)
    pad_top = (target_size - h) // 2
    pad_left = (target_size - w) // 2
    
    if pad_top >= 0 and pad_left >= 0 and (pad_top + h) <= target_size and (pad_left + w) <= target_size:
        canvas[pad_top : pad_top + h, pad_left : pad_left + w] = image
    
    return canvas

def apply_gradient_blur(image_array, channel_idx, p):
    original_dtype = image_array.dtype
    result_img = image_array.astype(np.float32)
    h, w, c = result_img.shape
    channel = result_img[:, :, channel_idx]
    sources = np.argwhere(channel > 0)
    if len(sources) == 0:
        return image_array 
    y_grid, x_grid = np.ogrid[-p:p+1, -p:p+1]
    dist_mask = np.sqrt(x_grid**2 + y_grid**2)
    base_mask = np.clip(1 - (dist_mask / p), 0, 1)
    for r_src, c_src in sources:
        source_value = channel[r_src, c_src]
        current_mask_scaled = base_mask * source_value
        r_min = max(0, r_src - p)
        r_max = min(h, r_src + p + 1)
        c_min = max(0, c_src - p)
        c_max = min(w, c_src + p + 1)
        mask_r_min = p - (r_src - r_min)
        mask_r_max = mask_r_min + (r_max - r_min)
        mask_c_min = p - (c_src - c_min)
        mask_c_max = mask_c_min + (c_max - c_min)
        mask_patch = current_mask_scaled[mask_r_min:mask_r_max, mask_c_min:mask_c_max]
        target_area = result_img[r_min:r_max, c_min:c_max, channel_idx]
        result_img[r_min:r_max, c_min:c_max, channel_idx] = np.maximum(target_area, mask_patch)
    if np.issubdtype(original_dtype, np.integer):
        result_img = np.clip(result_img, 0, 255)
    return result_img.astype(original_dtype)

def prepre_mask_for_learning(mask, higher_layer_masks, bboxes):
    combined = combine_parts_with_bbox(mask, higher_layer_masks, bboxes)
    removed_neighbours = remove_red_and_blue_with_red_neighbors(combined, d=6)
    resized_and_centered = resize_and_center(removed_neighbours)
    
    blured_mask = apply_gradient_blur(resized_and_centered[:, :, np.newaxis], channel_idx=0, p = 20)[:, :, 0]
    
    tensor_mask = prepare_single_channel_tensor(blured_mask)
    
    return tensor_mask

# --- 3. Core Logic  ---

def get_translation_matrix(first_frame_tensor, second_frame_tensor, prev_matrix=None):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tensor = first_frame_tensor.to(device)
    target_tensor = second_frame_tensor.to(device)

    # Initialize model with Warm Start
    model = AffineRegistrationModel(initial_theta=prev_matrix).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = OneSidedLoss()
    
    for i in range(150):
        optimizer.zero_grad()
        transformed_image = model(source_tensor)

        loss_coverage = criterion(transformed_image, target_tensor)

        loss_coverage.backward()
        optimizer.step()

    return model.theta.detach().cpu().numpy().tolist()

# --- 4. Other Logic ---

def combine_parts_with_bbox(blue_part, red_parts, bboxes):
    x_min = min(x for x, y, w, h in bboxes)
    y_min = min(y for x, y, w, h in bboxes)
    x_max = max(x + w for x, y, w, h in bboxes)
    y_max = max(y + h for x, y, w, h in bboxes)

    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    bx, by, bw, bh = bboxes[0]
    if blue_part is not None and blue_part.size > 0:
        canvas[by - y_min : by - y_min + blue_part.shape[0],
               bx - x_min : bx - x_min + blue_part.shape[1], 0] = blue_part

    for red_img, (rx, ry, rw, rh) in zip(red_parts, bboxes[1:]):
        if red_img is not None and red_img.size > 0:
            canvas[ry - y_min : ry - y_min + red_img.shape[0],
                   rx - x_min : rx - x_min + red_img.shape[1], 2] = red_img

    return canvas

def remove_red_and_blue_with_red_neighbors(img, d=1):
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    img = img.copy()
    B, G, R = cv2.split(img)
    
    red_mask = R > 0
    R[red_mask] = 0
    G[red_mask] = 0
    B[red_mask] = 0

    kernel_size = 2*d + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    red_dilated = binary_dilation(red_mask, structure=kernel)

    blue_mask = B > 0
    blue_to_remove = blue_mask & red_dilated
    B[blue_to_remove] = 0

    result = cv2.merge([B, G, R])
    
    coords = np.argwhere(np.any(result != 0, axis=2))
    if coords.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    cropped = result[y_min:y_max, x_min:x_max]
    
    blue_channel = cropped[:, :, 0]
    
    return blue_channel

def load_mask_from_some_frame(data, frame_index, part_index, atlas):
    
    mask_id = data["frames"][frame_index]["parts"][part_index]["mask_id"]
    
    return atlas.get_mask(mask_id, as_numpy=True)

def find_part_index_by_id(data,frame_index, part_id):
    for i, part in enumerate(data["frames"][frame_index]["parts"]):
        if part["id"] == part_id:
            return i
    return -1

def load_edge_masks_and_bboxes_from_frame(data, frame_index, part_ids, atlas):
    
    masks = []
    bboxes = []
                
    for i in part_ids:
        
        part_index = find_part_index_by_id(data, frame_index, i)
        
        if part_index == -1:
            continue
        
        h_mask = load_mask_from_some_frame(data, frame_index, part_index, atlas)
        
        if h_mask is not None:
            edge_mask = cv2.Canny(h_mask, 50, 150)
            masks.append(edge_mask)
            bboxes.append( data["frames"][frame_index]["parts"][part_index]["bbox"] )

    return masks, bboxes

def get_upper_layer_indices(element_id, layers):
    """
    Returns a list of indices of layers located above the layer 
    containing element_id.
    
    If the element is not found in layers, returns all indices (all layers are considered upper).
    """
    found_layer_index = -1

    # 1. Find which layer contains the element
    for i, layer in enumerate(layers):
        if element_id in layer:
            found_layer_index = i
            break
    
    # 2. Return logic
    if found_layer_index == -1:
        # If the element is not found, return indices of all layers (0, 1, 2...)
        return list(range(len(layers)))
    else:
        # If the element is found on layer i, return indices from (i + 1) to the end
        return list(range(found_layer_index + 1, len(layers)))

def get_data_from_specific_layers(all_layers_data, layer_indices):
    """
    Returns combined lists of masks and bboxes only for the specified layer indices.

    :param all_layers_data: List of data for all layers (where each element is (masks, bboxes)).
    :param layer_indices: List of layer indices to retrieve (e.g., [1, 2]).
    :return: (selected_masks, selected_bboxes)
    """
    selected_masks = []
    selected_bboxes = []

    for idx in layer_indices:
        # Check if the index is within array bounds
        if 0 <= idx < len(all_layers_data):
            # Unpack data for the specific layer
            layer_masks, layer_bboxes = all_layers_data[idx]
            
            # Add them to the overall result
            selected_masks.extend(layer_masks)
            selected_bboxes.extend(layer_bboxes)
        else:
            print(f"Warning: Layer with index {idx} does not exist.")

    return selected_masks, selected_bboxes
def check_scale_stability(new_matrix, prev_matrix, threshold=0.2):
    """
    Returns True if the scale change is acceptable (< threshold).
    Returns False if the change is too drastic.
    """
    if prev_matrix is None:
        return True

    # Extract matrix elements
    # New
    n_r1, n_r2 = new_matrix[0][0], new_matrix[0][1]
    n_r3, n_r4 = new_matrix[1][0], new_matrix[1][1]
    
    # Prev
    p_r1, p_r2 = prev_matrix[0][0], prev_matrix[0][1]
    p_r3, p_r4 = prev_matrix[1][0], prev_matrix[1][1]

    # Calculate vector magnitude (this represents the scale)
    # Scale X
    new_scale_x = (n_r1**2 + n_r2**2)**0.5
    prev_scale_x = (p_r1**2 + p_r2**2)**0.5
    
    # Scale Y
    new_scale_y = (n_r3**2 + n_r4**2)**0.5
    prev_scale_y = (p_r3**2 + p_r4**2)**0.5

    # Calculate average scale
    new_avg = (new_scale_x + new_scale_y) / 2.0
    prev_avg = (prev_scale_x + prev_scale_y) / 2.0

    # Protect against division by zero
    if prev_avg < 1e-6: 
        return True

    # Relative change
    delta = abs(new_avg - prev_avg) / prev_avg

    if delta > threshold:
        return False # Change too large
    
    return True # All good
def main():

    MAX_FRAMES_LIMIT = None

    frames = []

    parser = argparse.ArgumentParser(description="Texture Animator - Optimized Version")
    parser.add_argument("data_json")
    parser.add_argument("transform_json")
    parser.add_argument("--base-mesh-dir", default="visualization_base_mesh", help="Меш 1-го кадру з вагами")
    parser.add_argument("--mask-dir", default="visualization_transformed_masks", help="Деформовані білі маски")
    parser.add_argument("--out-dir", default="visualization_output", help="Дебаг оверлей (Зелене-Червоне)")
    parser.add_argument("--test-id", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.data_json , 'r') as f:
        data = json.load(f)
        
        layers = data['layers']
        non_drawable_parts = data['non_drawable_part_ids']

        atlas = AtlasManager(data, Path(args.data_json).parent)
        
        first_frame_tensors = []
        first_frame_id = []
        number_of_parts = len( data["frames"][0]["parts"] )
        
        # Dictionary to store matrices from the previous frame
        last_matrices = {}
        
        layers_masks_and_bboxes = []
        
        for layer in layers:
            layers_masks_and_bboxes.append(load_edge_masks_and_bboxes_from_frame(data, 0, layer, atlas))

        for p in range(1, data["total_unique_part_ids"] + 1):
            
            if p in non_drawable_parts:
                continue
            
            part_index = find_part_index_by_id(data, 0, p)
            
            if part_index == -1:
                continue
            
            mask = load_mask_from_some_frame(data, 0, part_index, atlas)
            edge_mask = cv2.Canny(mask, 50, 150)
            
            if edge_mask is None:
                    continue
            
            current_tensor = None
            
            upper_layers = get_upper_layer_indices(p, layers)
            
            if len(upper_layers) != 0:
                higher_edge_masks, higher_bboxes = get_data_from_specific_layers(layers_masks_and_bboxes, upper_layers)
                current_tensor = prepre_mask_for_learning(edge_mask, higher_edge_masks,
                                                          [ data["frames"][0]["parts"][part_index]["bbox"] ] + higher_bboxes)
            else:
                current_tensor = prepre_mask_for_learning(edge_mask, [],  [ data["frames"][0]["parts"][part_index]["bbox"] ])

            first_frame_tensors.append(current_tensor)
            
            curr_id = p
            first_frame_id.append(curr_id)
            
            # Initialize Identity matrix for the first frame
            last_matrices[curr_id] = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ]

        total_frames = len(data["frames"])
        frames_to_process = total_frames
        
        if MAX_FRAMES_LIMIT is not None:
            frames_to_process = min(total_frames, MAX_FRAMES_LIMIT)
            print(f"Limiting processing to first {frames_to_process} frames.")

        for i in range(1, frames_to_process):
            parts_transform = []
            print(f"Processing frame {i} / {frames_to_process - 1}")
            
            layers_masks_and_bboxes = []
        
            for layer in layers:
                layers_masks_and_bboxes.append(load_edge_masks_and_bboxes_from_frame(data, i, layer, atlas))

            for j in range(1, data["total_unique_part_ids"] + 1):
                
                if j in non_drawable_parts:
                    continue

                if j not in first_frame_id:
                    continue
                
                part_index = find_part_index_by_id(data, i, j)
                
                if part_index == -1:
                    continue
                
                mask = load_mask_from_some_frame(data, i, part_index, atlas)
                
                if mask is None:
                    continue
                    
                current_edge_mask = cv2.Canny(mask, 50, 150)
                if current_edge_mask is None:
                    continue
                
                current_tensor = None

                upper_layers = get_upper_layer_indices(j, layers)

                if len(upper_layers) != 0:
                    
                    higher_edge_masks, higher_bboxes = get_data_from_specific_layers(layers_masks_and_bboxes, upper_layers)
                    
                    current_tensor = prepre_mask_for_learning(current_edge_mask, higher_edge_masks,
                                                            [ data["frames"][i]["parts"][part_index]["bbox"] ] + higher_bboxes)
                else:
                    current_tensor = prepre_mask_for_learning(current_edge_mask, [],  [ data["frames"][i]["parts"][part_index]["bbox"] ])

                current_id = j
                index_in_first = first_frame_id.index(current_id)

                # Get the matrix from the previous frame
                prev_matrix = last_matrices.get(current_id, None)

                transform_matrix = get_translation_matrix(
                    first_frame_tensors[index_in_first], 
                    current_tensor,
                    prev_matrix=prev_matrix
                )
                
                if prev_matrix is not None:
                    # Check if the scale change is stable (threshold 7%)
                    is_stable = check_scale_stability(transform_matrix, prev_matrix, threshold=0.07)
                    
                    if not is_stable:
                        # If change > 7%, ignore the new matrix and revert to the previous one
                        # print(f"Warning: Frame {i}, Part {current_id} - Scale jump detected. Reverting.")
                        transform_matrix = prev_matrix
                
                # Update history
                last_matrices[current_id] = transform_matrix
                
                parts_transform.append({"id":current_id, "matrix":transform_matrix})
                
            frames.append(parts_transform)
    
    json.dump(
        frames, 
        open("transform.json", 'w', encoding='utf-8'), 
        ensure_ascii=False,
        indent=4
    )

if __name__ == "__main__":
    main()