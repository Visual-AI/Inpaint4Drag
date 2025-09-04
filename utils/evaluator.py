"""
Module for evaluating image editing with stable diffusion features and perceptual metrics.
Includes functionality for feature extraction and similarity measurements.
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import lpips
from diffusers import StableDiffusionPipeline

# Suppress UserWarnings
warnings.filterwarnings(action='ignore', category=UserWarning)

class SDFeaturizer(StableDiffusionPipeline):
    """
    Extracts Stable Diffusion 2.1 features for semantic point matching (DIFT).
    Inherits from StableDiffusionPipeline and adds feature extraction capabilities.
    """
    
    @torch.no_grad()
    def __call__(
        self, 
        img_tensor: torch.Tensor, 
        t: int = 261, 
        ensemble: int = 8, 
        prompt: str = None, 
        prompt_embeds: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract features from input image tensor.
        
        Args:
            img_tensor: Input image tensor (B,C,H,W)
            t: Timestep for noise addition
            ensemble: Number of ensemble predictions
            prompt: Text prompt for conditioning
            prompt_embeds: Pre-computed prompt embeddings
            
        Returns:
            torch.Tensor: Extracted features
        """
        assert img_tensor.shape[0] == 1, "Batch size must be 1"
        device = self._execution_device
        
        # Encode image to latent space
        latents = self.vae.encode(img_tensor).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.expand(ensemble, -1, -1, -1)
        
        # Add noise
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # Get prompt embeddings
        if prompt_embeds is None:
            prompt = "" if prompt is None else prompt
            prompt_embeds = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )[0]
        prompt_embeds = prompt_embeds.expand(ensemble, -1, -1)

        # Extract features using forward hook
        unet_feature = []
        def hook(module, input, output):
            unet_feature.clear()
            unet_feature.append(output)
            
        handle = list(self.unet.children())[4][1].register_forward_hook(hook=hook)
        self.unet(latents_noisy, t, prompt_embeds)
        handle.remove()

        return unet_feature[0].mean(dim=0, keepdim=True)

class DragEvaluator:
    """
    Evaluator for computing perceptual and distance metrics between images.
    
    Provides methods for LPIPS similarity and point-based distance measurements.
    """
    
    def __init__(self):
        """Initialize the evaluator with required models and settings."""
        self.sd_loaded = False
        self.lpips_loaded = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16

    def load_sd(self):
        """Load Stable Diffusion model if not already loaded."""
        if not self.sd_loaded:
            self.sd_feat = SDFeaturizer.from_pretrained(
                'stabilityai/stable-diffusion-2-1', 
                torch_dtype=self.dtype
            ).to(self.device)
            self.sd_loaded = True

    def load_lpips(self):
        """Load LPIPS model if not already loaded."""
        if not self.lpips_loaded:
            self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device).to(self.dtype)
            self.lpips_loaded = True

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor and normalize to [-1, 1].
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        return image.to(self.device).to(self.dtype)
    
    @torch.no_grad()
    def compute_lpips(self, original_image: np.ndarray, edited_image: np.ndarray) -> float:
        """
        Compute LPIPS perceptual similarity between two images.
        
        Args:
            original_image: Original image array
            edited_image: Edited image array
            
        Returns:
            float: LPIPS similarity score
        """
        self.load_lpips()
        
        image1 = F.interpolate(self.preprocess_image(original_image), (224, 224), mode='bilinear')
        image2 = F.interpolate(self.preprocess_image(edited_image), (224, 224), mode='bilinear')

        return self.loss_fn_alex(image1, image2).item()

    @torch.no_grad()
    def compute_distance(self, original_image: np.ndarray, edited_image: np.ndarray, 
                        handle_pts: np.ndarray, target_pts: np.ndarray, 
                        prompt: str = None, plot_path: str = None) -> float:
        """
        Compute mean distance metric between handle and target points.
        
        Args:
            original_image: Original image array
            edited_image: Edited image array
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            prompt: Optional text prompt
            plot_path: Optional path to save visualization
            
        Returns:
            float: Mean distance metric
        """
        self.load_sd()
        
        handle_pts = torch.tensor(handle_pts, device=self.device, dtype=torch.long)
        target_pts = torch.tensor(target_pts, device=self.device, dtype=torch.long)
        
        # Handle image size mismatch
        if original_image.shape != edited_image.shape:
            orig_h, orig_w = original_image.shape[:2]
            edit_h, edit_w = edited_image.shape[:2]
            edited_image = cv2.resize(edited_image, (orig_w, orig_h))
            target_pts = target_pts * torch.tensor([orig_w, orig_h], device=self.device)
            target_pts = (target_pts / torch.tensor([edit_w, edit_h], device=self.device)).long()

        image_h, image_w = original_image.shape[:2]
        orig_img = F.interpolate(self.preprocess_image(original_image), size=(768, 768))
        edit_img = F.interpolate(self.preprocess_image(edited_image), size=(768, 768))

        # Extract and process features
        orig_feat = F.interpolate(self.sd_feat(orig_img, prompt=prompt), size=(image_h, image_w))
        edit_feat = F.interpolate(self.sd_feat(edit_img, prompt=prompt), size=(image_h, image_w))

        mask = self._create_mask(handle_pts, target_pts, (image_h, image_w))
        matched_pts = self._nn_get_matches(orig_feat, edit_feat, handle_pts, mask=mask)
        
        # Calculate distance metric
        dist = target_pts - matched_pts
        dist = dist.float() / torch.tensor([image_w, image_h], device=self.device)
        mean_dist = dist.norm(dim=-1).mean().item()

        if plot_path:
            self.plot_drag_result(
                edited_image,
                matched_pts.cpu().numpy(),
                target_pts.cpu().numpy(),
                output_path=plot_path
            )

        return mean_dist

    @staticmethod
    def plot_drag_result(edited_image: np.ndarray, handle_pts: np.ndarray, 
                        target_pts: np.ndarray, output_path: str = None):
        """
        Plot drag editing results with arrows showing point movements.
        
        Args:
            edited_image: Edited image array
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            output_path: Optional path to save visualization
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(edited_image)
        
        # Convert points to numpy if needed
        if torch.is_tensor(handle_pts):
            handle_pts = handle_pts.cpu().numpy()
        if torch.is_tensor(target_pts):
            target_pts = target_pts.cpu().numpy()
        
        # Plot points and arrows
        plt.scatter(target_pts[:, 0], target_pts[:, 1], c='blue', label='Target Points')
        plt.scatter(handle_pts[:, 0], handle_pts[:, 1], c='red', label='Matched Points')
        
        for i in range(len(handle_pts)):
            plt.arrow(handle_pts[i, 0], handle_pts[i, 1],
                     target_pts[i, 0] - handle_pts[i, 0],
                     target_pts[i, 1] - handle_pts[i, 1],
                     color='white', head_width=5, head_length=5)
        
        plt.legend()
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _create_mask(handle_pts: torch.Tensor, target_pts: torch.Tensor, 
                     img_size: tuple) -> torch.Tensor:
        """
        Create masks based on pixel distances to point pairs.
        
        Args:
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            img_size: Image dimensions (H,W)
            
        Returns:
            torch.Tensor: Binary mask
        """
        handle_pts, target_pts = handle_pts.float(), target_pts.float()
        h, w = img_size
        
        min_dist = ((handle_pts - target_pts).norm(dim=1) / 2**0.5).clamp(min=5)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=handle_pts.device),
            torch.arange(w, device=handle_pts.device),
            indexing="ij"
        )
        
        y_grid = y_grid.expand(len(handle_pts), -1, -1)
        x_grid = x_grid.expand(len(handle_pts), -1, -1)
        
        handle_dist = ((x_grid - handle_pts[:, None, None, 0])**2 + (y_grid - handle_pts[:, None, None, 1])**2).sqrt()
        target_dist = ((x_grid - target_pts[:, None, None, 0])**2 + (y_grid - target_pts[:, None, None, 1])**2).sqrt()
        
        return (handle_dist < min_dist[:, None, None]) | (target_dist < min_dist[:, None, None])

    @staticmethod
    def _nn_get_matches(src_featmaps: torch.Tensor, trg_featmaps: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Find nearest neighbor matches between source and target feature maps.
        
        Args:
            src_featmaps: Source feature maps
            trg_featmaps: Target feature maps
            query: Query points
            l2_norm: Whether to apply L2 normalization
            mask: Optional mask for valid matches
            
        Returns:
            torch.Tensor: Matched point coordinates
        """
        _, c, h, w = src_featmaps.shape
        query = query.long()
        src_feat = src_featmaps[0, :, query[:, 1], query[:, 0]]
        src_feat = F.normalize(src_feat, p=2, dim=0)

        trg_featmaps = F.normalize(trg_featmaps, p=2, dim=1)
        trg_featmaps = trg_featmaps.view(c, -1)
        similarity = torch.mm(src_feat.t(), trg_featmaps)
        
        if mask is not None:
            similarity = torch.where(
                mask.view(-1, h * w), 
                similarity, 
                torch.full_like(similarity, -torch.inf)
            )
        
        best_idx = similarity.argmax(dim=-1)
        y_coords = best_idx // w
        x_coords = best_idx % w
        
        return torch.stack((x_coords, y_coords), dim=1).float()