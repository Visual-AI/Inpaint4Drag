import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


def download_model(checkpoint_path: str, model_name: str = "efficientvit_sam_l0.pt") -> str:
    """
    Download the model checkpoint if not found locally.

    Args:
        checkpoint_path: Local path where model should be saved
        model_name: Name of the model file to download

    Returns:
        str: Path to the downloaded checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    base_url = "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main"
    model_url = f"{base_url}/{model_name}"
    
    try:
        print(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, checkpoint_path)
        print(f"Model successfully downloaded to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {str(e)}")


class SamMaskRefiner(nn.Module):
    CHECKPOINT_DIR = 'checkpoints'
    MODEL_CONFIGS = {
        'l0': 'efficientvit_sam_l0.pt',
        'l1': 'efficientvit_sam_l1.pt',
        'l2': 'efficientvit_sam_l2.pt'
    }
    
    def __init__(self, model_name: str = 'l0') -> None:
        """
        Initialize SAM predictor with specified model version.

        Args:
            model_name: Model version to use ('l0', 'l1', or 'l2'). Defaults to 'l0'.

        Raises:
            ValueError: If invalid model_name is provided
            RuntimeError: If model loading fails after download attempt
        """
        super().__init__()
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_name. Choose from: {list(self.MODEL_CONFIGS.keys())}")
            
        model_filename = self.MODEL_CONFIGS[model_name]
        checkpoint_path = os.path.join(self.CHECKPOINT_DIR, model_filename)
        
        try:
            from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
            from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        except ImportError:
            raise ImportError(
                "Failed to import EfficientViT modules. Please ensure the package is installed:\n"
                "pip install git+https://github.com/mit-han-lab/efficientvit.git"
            )
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Attempting to download...")
            checkpoint_path = download_model(checkpoint_path, model_filename)
        
        try:
            model_type = f'efficientvit-sam-{model_name}'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = create_efficientvit_sam_model(model_type, True, checkpoint_path).eval()
            self.model = self.model.requires_grad_(False).to(device)
            self.predictor = EfficientViTSamPredictor(self.model)
            print(f"\033[92mEfficientViT-SAM model loaded from: {checkpoint_path}\033[0m")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def sample_points_from_mask(self, mask: np.ndarray, max_points: int = 128) -> np.ndarray:
        """
        Sample points uniformly from masked regions.

        Args:
            mask: Binary mask array of shape (H, W) with 0-1 values.
            max_points: Maximum number of points to sample.

        Returns:
            np.ndarray: Array of shape (N, 2) containing [x,y] coordinates.
        """
        y_indices, x_indices = np.where(mask > 0.5)
        total_points = len(y_indices)
        
        if total_points <= max_points:
            return np.stack([x_indices, y_indices], axis=1)
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        aspect_ratio = (x_max - x_min) / max(y_max - y_min, 1)
        ny = int(np.sqrt(max_points / aspect_ratio))
        nx = int(ny * aspect_ratio)
        
        x_bins = np.linspace(x_min, x_max + 1, nx + 1, dtype=np.int32)
        y_bins = np.linspace(y_min, y_max + 1, ny + 1, dtype=np.int32)
        
        x_dig = np.digitize(x_indices, x_bins) - 1
        y_dig = np.digitize(y_indices, y_bins) - 1
        bin_indices = y_dig * nx + x_dig
        unique_bins = np.unique(bin_indices)
        
        points = []
        for idx in unique_bins:
            bin_y = idx // nx
            bin_x = idx % nx
            mask = (y_dig == bin_y) & (x_dig == bin_x)
            
            if np.any(mask):
                px = int(np.mean(x_indices[mask]))
                py = int(np.mean(y_indices[mask]))
                points.append([px, py])
        
        points = np.array(points)
        
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[indices]
        
        return points

    def refine_mask(self, image: np.ndarray, input_mask: np.ndarray, kernel_size: int = 21) -> np.ndarray:
        """
        Refine an input mask using the SAM (Segment Anything Model) model.

        Args:
            image: RGB image, shape (H, W, 3), values in [0, 255]
            input_mask: Binary mask, shape (H, W), values in {0, 1}
            kernel_size: Size of morphological kernel (default: 21)

        Returns:
            Refined binary mask, shape (H, W), values in {0, 1}
        """
        points = self.sample_points_from_mask(input_mask, max_points=128)
        if len(points) == 0:
            return input_mask

        self.predictor.set_image(image)
        masks_pred, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=np.ones(len(points)),
            multimask_output=False
        )
        sam_mask = masks_pred[0]

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded_input = cv2.dilate(input_mask.astype(np.uint8), kernel)
        preserved_input = cv2.erode(input_mask.astype(np.uint8), kernel)
    
        sam_mask = np.logical_and(expanded_input, sam_mask).astype(np.uint8)
        sam_mask = np.logical_or(preserved_input, sam_mask).astype(np.uint8)
        
        return sam_mask