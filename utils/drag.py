import numpy as np
import cv2
import torch
from typing import Union

def contour_to_points_and_mask(contour: np.ndarray, image_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Convert a contour to a set of points and binary mask.
    
    This function takes a contour and creates both a binary mask and a list of points
    that lie within the contour. The points are represented in (x, y) coordinates.
    
    Args:
        contour (np.ndarray): Input contour of shape (N, 2) or (N, 1, 2) where N is 
            the number of points. Each point should be in (x, y) format.
        image_shape (tuple): Shape of the output mask as (height, width).
            
    Returns:
        tuple:
            - np.ndarray: Array of points in (x, y) format with shape (M, 2),
              where M is the number of points inside the contour.
              Returns empty array of shape (0, 2) if contour is empty.
            - np.ndarray: Binary mask of shape image_shape where pixels inside
              the contour are 255 and outside are 0.
    """
    if len(contour) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(image_shape, dtype=np.uint8)
    
    # Create empty mask and fill the contour in the mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour.reshape(-1, 1, 2)], -1, 255, cv2.FILLED)
    
    # Get points inside contour (y, x) and convert to (x, y)
    points = np.column_stack(np.where(mask)).astype(np.int32)[:, [1, 0]]
    
    # Return empty array if no points found
    if len(points) == 0:
        points = np.zeros((0, 2), dtype=np.int32)
        
    return points, mask

def find_control_points(
    region_points: torch.Tensor,
    source_control_points: torch.Tensor,
    target_control_points: torch.Tensor,
    distance_threshold: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find control points that match points within a region.

    This function identifies which control points lie within or very close to
    the specified region points. It matches source control points to region points
    and returns both source and corresponding target control points that satisfy
    the distance threshold criterion.

    Args:
        region_points (torch.Tensor): Points defining a region, shape (N, 2).
            Each point is in (x, y) format.
        source_control_points (torch.Tensor): Source control points, shape (M, 2).
            Each point is in (x, y) format.
        target_control_points (torch.Tensor): Target control points, shape (M, 2).
            Must have same first dimension as source_control_points.
        distance_threshold (float, optional): Maximum distance for a point to be
            considered matching. Defaults to 1e-6.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - Matched source control points, shape (K, 2) where K ≤ M
            - Corresponding target control points, shape (K, 2)
            If no matches found or inputs empty, returns empty tensors of shape (0, 2)
    """
    # Handle empty input cases
    if len(region_points) == 0 or len(source_control_points) == 0:
        return (
            torch.zeros((0, 2), device=source_control_points.device),
            torch.zeros((0, 2), device=target_control_points.device)
        )

    # Calculate pairwise distances between source control points and region points
    distances = torch.cdist(source_control_points, region_points)
    
    # Find points that are within threshold distance of any region point
    min_distances = distances.min(dim=1)[0]
    matching_indices = min_distances < distance_threshold

    # Return matched pairs of control points
    return source_control_points[matching_indices], target_control_points[matching_indices]

def interpolate_points_with_weighted_directions(
    points: torch.Tensor,
    reference_points: torch.Tensor,
    direction_vectors: torch.Tensor,
    max_reference_points: int = 100,
    num_nearest_neighbors: int = 4,
    eps: float = 1e-6
) -> torch.Tensor:
    """Interpolate points based on weighted directions from nearest reference points.
    
    This function moves each point by a weighted combination of direction vectors.
    The weights are determined by the inverse distances to the nearest reference points.
    If there are too many reference points, they are subsampled for efficiency.

    Args:
        points (torch.Tensor): Points to interpolate, shape (N, 2) in (x, y) format
        reference_points (torch.Tensor): Reference point locations, shape (M, 2)
        direction_vectors (torch.Tensor): Direction vectors for each reference point,
            shape (M, 2), must match reference_points first dimension
        max_reference_points (int, optional): Maximum number of reference points to use.
            If exceeded, points are subsampled. Defaults to 100.
        num_nearest_neighbors (int, optional): Number of nearest neighbors to consider
            for interpolation. Defaults to 4.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Interpolated points with shape (N, 2). If input points or
            references are empty, returns the input points unchanged.
    """
    # Handle empty input cases
    if len(points) == 0 or len(reference_points) == 0:
        return points
    
    # Handle single reference point case
    if len(reference_points) == 1:
        return points + direction_vectors

    # Subsample reference points if too many
    if len(reference_points) > max_reference_points:
        indices = torch.linspace(0, len(reference_points)-1, max_reference_points).long()
        reference_points = reference_points[indices]
        direction_vectors = direction_vectors[indices]

    # Calculate distances to all reference points
    distances = torch.cdist(points, reference_points)
    
    # Find k nearest neighbors (k = min(num_nearest_neighbors, num_references))
    k = min(num_nearest_neighbors, len(reference_points))
    topk_distances, neighbor_indices = torch.topk(
        distances, 
        k=k, 
        dim=1, 
        largest=False
    )
    
    # Calculate weights based on inverse distances
    weights = 1.0 / (topk_distances + eps)
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # Get directions for nearest neighbors and compute weighted average
    neighbor_directions = direction_vectors[neighbor_indices]
    weighted_directions = (weights.unsqueeze(-1) * neighbor_directions).sum(dim=1)
    
    # Apply weighted directions and round to nearest integer
    interpolated_points = (points + weighted_directions).round().float()
    
    return interpolated_points

def get_points_within_image_bounds(
    points: torch.Tensor,
    image_shape: tuple[int, int]
) -> torch.Tensor:
    """Create a boolean mask for points that lie within image boundaries.

    Identifies which points from the input tensor fall within valid image coordinates.
    Points are assumed to be in (x, y) format, while image_shape is in (height, width) format.

    Args:
        points (torch.Tensor): Points to check, shape (N, 2) in (x, y) format.
            x coordinates correspond to width/columns
            y coordinates correspond to height/rows
        image_shape (tuple[int, int]): Image dimensions as (height, width).

    Returns:
        torch.Tensor: Boolean mask of shape (N,) where True indicates the point
            is within bounds. Returns empty tensor of shape (0,) if input is empty.
    """
    # Handle empty input case
    if len(points) == 0:
        return torch.zeros(0, dtype=torch.bool, device=points.device)

    # Unpack image dimensions
    height, width = image_shape

    # Check both x and y coordinates are within bounds
    x_in_bounds = (points[:, 0] >= 0) & (points[:, 0] < width)
    y_in_bounds = (points[:, 1] >= 0) & (points[:, 1] < height)
    
    # Combine conditions
    valid_points_mask = x_in_bounds & y_in_bounds

    return valid_points_mask

def bi_warp(
    region_mask: np.ndarray,
    control_points: Union[np.ndarray, torch.Tensor],
    kernel_size: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate corresponding source/target points and inpainting mask for masked regions.
    
    Args:
        region_mask: Binary mask defining regions of interest (2D array with 0s and 1s)
        control_points: Alternating source and target control points. Shape (N*2, 2)
        kernel_size: Controls dilation kernel size. Must be odd number or 0.
                    Contour thickness will be (kernel_size-1)*2 (default: 5)
                    Set to 0 for no contour drawing and no dilation.
    
    Returns:
        tuple containing:
            - Source points (M, 2)
            - Target points (M, 2) 
            - Inpainting mask combined with target contour mask
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_shape = region_mask.shape
    
    # Ensure kernel_size is odd or 0
    kernel_size = max(0, kernel_size)
    if kernel_size > 0 and kernel_size % 2 == 0:
        kernel_size += 1

    # 1. Initialize tensors and masks
    control_points = torch.tensor(control_points, dtype=torch.float32, device=device) if not isinstance(control_points, torch.Tensor) else control_points
    source_control_points = control_points[0:-1:2]
    target_control_points = control_points[1::2]
    
    combined_source_mask = np.zeros(image_shape, dtype=np.uint8)
    combined_target_mask = np.zeros(image_shape, dtype=np.uint8)
    region_mask_binary = np.where(region_mask > 0, 1, 0).astype(np.uint8)
    contour_mask = np.zeros(image_shape, dtype=np.uint8)

    # 2. Process regions
    contours = cv2.findContours(region_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    all_source_points = []
    all_target_points = []

    for contour in contours:
        if len(contour) == 0:
            continue

        # 3. Get source region points and mask
        source_contour = torch.from_numpy(contour[:, 0, :]).float().to(device)
        source_region_points, source_mask = contour_to_points_and_mask(contour[:, 0, :], image_shape)
        source_mask = (source_mask > 0).astype(np.uint8)
        
        if len(source_region_points) == 0:
            continue
            
        source_region_points = torch.from_numpy(source_region_points).float().to(device)

        # 4. Transform points
        source, target = find_control_points(source_region_points, source_control_points, target_control_points)
        if len(source) == 0:
            continue

        directions = target - source
        target_contour = interpolate_points_with_weighted_directions(source_contour, source, directions)
        interpolated_target = interpolate_points_with_weighted_directions(source_region_points, source, directions)

        # 5. Get target region points and mask
        target_region_points, target_mask = contour_to_points_and_mask(target_contour.cpu().int().numpy(), image_shape)
        target_mask = (target_mask > 0).astype(np.uint8)
        
        if len(target_region_points) == 0:
            continue
            
        # Draw target contour
        target_contour_np = target_contour.cpu().int().numpy()
        if kernel_size > 0:
            cv2.drawContours(contour_mask, [target_contour_np], -1, 1, kernel_size)
            
        target_region = torch.from_numpy(target_region_points).float().to(device)

        # 6. Apply reverse transformation
        back_directions = source_region_points - interpolated_target
        interpolated_source = interpolate_points_with_weighted_directions(target_region, interpolated_target, back_directions)

        # 7. Filter valid points
        valid_mask = get_points_within_image_bounds(interpolated_source, image_shape)
        if valid_mask.any():
            all_source_points.append(interpolated_source[valid_mask])
            all_target_points.append(target_region[valid_mask])
            combined_source_mask = np.logical_or(combined_source_mask, source_mask).astype(np.uint8)
            combined_target_mask = np.logical_or(combined_target_mask, target_mask).astype(np.uint8)

    # 8. Handle empty case
    if not all_source_points:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32), np.zeros(image_shape, dtype=np.uint8)

    # 9. Finalize outputs
    final_source = torch.cat(all_source_points).cpu().numpy().astype(np.int32)
    final_target = torch.cat(all_target_points).cpu().numpy().astype(np.int32)
    
    # Create and combine masks
    inpaint_mask = np.logical_and(combined_source_mask, np.logical_not(combined_target_mask)).astype(np.uint8)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel)
    final_mask = np.logical_or(inpaint_mask, contour_mask).astype(np.uint8)

    return final_source, final_target, final_mask