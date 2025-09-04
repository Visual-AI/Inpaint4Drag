import os
import pickle
from time import perf_counter

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting, AutoencoderTiny, LCMScheduler

from utils.drag import bi_warp
from utils.refine_mask import SamMaskRefiner


__all__ = [
    'clear_all', 'resize',
    'visualize_user_drag', 'preview_out_image', 'inpaint',
    'add_point', 'undo_point', 'clear_point',
]

# UI functions
def clear_all(length):
    """Reset UI by clearing all input images and parameters."""
    return (gr.Image(value=None, height=length, width=length),) * 3 + ([], 21, 5, "output/app", None)

def resize(canvas, gen_length, canvas_length):
    """Resize canvas while maintaining aspect ratio."""
    if not canvas:
        return (gr.Image(value=None, width=canvas_length, height=canvas_length),) * 3

    image = process_canvas(canvas)[0]
    aspect_ratio = image.shape[1] / image.shape[0]
    is_landscape = aspect_ratio >= 1

    new_dims = (
        (gen_length, round(gen_length / aspect_ratio / 8) * 8) if is_landscape
        else (round(gen_length * aspect_ratio / 8) * 8, gen_length)
    )
    canvas_dims = (
        (canvas_length, round(canvas_length / aspect_ratio)) if is_landscape
        else (round(canvas_length * aspect_ratio), canvas_length)
    )

    return (gr.Image(value=cv2.resize(image, new_dims), width=canvas_dims[0], height=canvas_dims[1]),) * 3

def process_canvas(canvas):
    """Extracts the image (H, W, 3) and the mask (H, W) from a Gradio canvas object."""
    image = canvas["image"].copy()
    mask = np.uint8(canvas["mask"][:, :, 0] > 0).copy()
    return image, mask

# Point manipulation functions
def add_point(canvas, points, sam_ks, if_sam, output_path, evt: gr.SelectData):
    """Add selected point to points list and update image."""
    if canvas is None:
        return None 
    points.append(evt.index)
    return visualize_user_drag(canvas, points, sam_ks, if_sam, output_path)

def undo_point(canvas, points, sam_ks, if_sam, output_path):
    """Remove last point and update image."""
    if canvas is None:
        return None 
    if len(points) > 0:
        points.pop()
    return visualize_user_drag(canvas, points, sam_ks, if_sam, output_path)

def clear_point(canvas, points, sam_ks, if_sam, output_path):
    """Clear all points and update image."""
    if canvas is None:
        return None 
    points.clear()
    return visualize_user_drag(canvas, points, sam_ks, if_sam, output_path)

# Visualization tools
def refine_mask(image, mask, kernel_size):
    """Refine mask using SAM model if available."""
    global sam_refiner
    try:
        if 'sam_refiner' not in globals():
            sam_refiner = SamMaskRefiner()
        return sam_refiner.refine_mask(image, mask, kernel_size)
    except ImportError:
        gr.Warning("EfficientVit not installed. Please install with: pip install git+https://github.com/mit-han-lab/efficientvit.git")
        return mask
    except Exception as e:
        gr.Warning(f"Error refining mask: {str(e)}")
        return mask

def visualize_user_drag(canvas, points, sam_ks, if_sam=False, output_path=None):
    """Visualize control points and motion vectors on the input image.
    
    Args:
        canvas (dict): Gradio canvas containing image and mask
        points (list): List of (x,y) coordinate pairs for control points
        sam_ks (int): Kernel size for SAM mask refinement
        if_sam (bool): Whether to use SAM refinement on mask
    """
    if canvas is None:
        return None
    
    image, mask = process_canvas(canvas)
    mask = refine_mask(image, mask, sam_ks) if if_sam and mask.sum() > 0 else mask

    # Apply colored mask overlay
    result = image.copy()
    result[mask == 1] = [255, 0, 0]  # Red color
    image = cv2.addWeighted(result, 0.3, image, 0.7, 0)
    
    # Draw mask outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
    
    # Draw control points and motion vectors
    for idx, point in enumerate(points, 1):
        if idx % 2 == 0:
            cv2.circle(image, tuple(point), 10, (0, 0, 255), -1)  # End point
            cv2.arrowedLine(image, prev_point, point, (255, 255, 255), 4, tipLength=0.5)
        else:
            cv2.circle(image, tuple(point), 10, (255, 0, 0), -1)  # Start point
            prev_point = point

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        Image.fromarray(image).save(os.path.join(output_path, 'user_drag_i4p.png'))
    return image

def preview_out_image(canvas, points, sam_ks, inpaint_ks, if_sam=False, output_path=None):
    """Preview warped image result and generate inpainting mask.
    
    Args:
        canvas (dict): Gradio canvas containing the input image and mask
        points (list): List of (x,y) coordinate pairs defining source and target positions for warping
        sam_ks (int): Kernel size parameter for SAM mask refinement
        inpaint_ks (int): Kernel size parameter for inpainting mask generation
        if_sam (bool): Whether to use SAM model for mask refinement
        output_path (str, optional): Directory path to save original image and metadata
        
    Returns:
        tuple: 
            - ndarray: Warped image with grid pattern overlay on regions needing inpainting
            - ndarray: Binary mask (255 for inpainting regions, 0 elsewhere)
            - (None, None): If canvas is empty or fewer than 2 control points provided
    """
    if canvas is None:
        return None, None
    
    image, mask = process_canvas(canvas)
    if len(points) < 2:
        return image, None
    
    # ensure H, W divisible by 8 and longer edge 512
    shapes_valid = all(s % 8 == 0 for s in mask.shape + image.shape[:2])
    size_valid = all(max(x.shape[:2] if len(x.shape) > 2 else x.shape) == 512 for x in (image, mask))
    if not (shapes_valid and size_valid):
        gr.Warning('Click Resize Image Button first.')

    mask = refine_mask(image, mask, sam_ks) if if_sam and mask.sum() > 0 else mask
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        Image.fromarray(image).save(os.path.join(output_path, 'original_image.png'))
        metadata = {'mask': mask, 'points': points}
        with open(os.path.join(output_path, 'meta_data_i4p.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
    
    handle_pts, target_pts, inpaint_mask = bi_warp(mask, points, inpaint_ks)
    image[target_pts[:, 1], target_pts[:, 0]] = image[handle_pts[:, 1], handle_pts[:, 0]]

    # Add grid pattern to highlight inpainting regions
    background = np.ones_like(mask) * 255
    background[::10] = background[:, ::10] = 0
    image = np.where(inpaint_mask[..., np.newaxis]==1, background[..., np.newaxis], image)

    if output_path:
        Image.fromarray(image).save(os.path.join(output_path, 'preview_image.png'))
    
    return image, (inpaint_mask * 255).astype(np.uint8)

# Inpaint tools
def setup_pipeline(device='cuda', model_version='v1-5'):
    """Initialize optimized inpainting pipeline with specified model configuration."""
    MODEL_CONFIGS = {
        'v1-5': ('runwayml/stable-diffusion-inpainting', 'latent-consistency/lcm-lora-sdv1-5', 'madebyollin/taesd'),
        'xl': ('diffusers/stable-diffusion-xl-1.0-inpainting-0.1', 'latent-consistency/lcm-lora-sdxl', 'madebyollin/taesdxl')
    }
    model_id, lora_id, vae_id = MODEL_CONFIGS[model_version]

    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_id)
    pipe.fuse_lora()
    pipe.vae = AutoencoderTiny.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    # Pre-compute prompt embeddings during setup
    if model_version == 'v1-5':
        pipe.cached_prompt_embeds = pipe.encode_prompt(
            '', device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False)[0]
    else:
        pipe.cached_prompt_embeds, pipe.cached_pooled_prompt_embeds = pipe.encode_prompt(
            '', device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False)[0::2]
            
    return pipe

pipe = setup_pipeline(model_version='v1-5')
pipe.cached_prompt_embeds = pipe.encode_prompt('', 'cuda', 1, False)[0]

def inpaint(image, inpaint_mask):
    """Perform efficient inpainting on masked regions using Stable Diffusion.
    
    Args:
        image (ndarray): Input RGB image array (warped preview image)
        inpaint_mask (ndarray): Binary mask array where 255 indicates regions to inpaint
        
    Returns:
        ndarray: Inpainted image with masked regions filled in
    """
    if image is None:
        return None

    if inpaint_mask is None:
        return image
    
    start = perf_counter()
    pipe_id = 'xl' if 'xl' in pipe.config._name_or_path else 'v1-5'
    inpaint_strength = 0.99 if pipe_id == 'xl' else 1.0

    # Convert inputs to PIL
    image_pil = Image.fromarray(image)
    inpaint_mask_pil = Image.fromarray(inpaint_mask) 

    width, height = inpaint_mask_pil.size
    if width % 8 != 0 or height % 8 != 0:
        width, height = round(width / 8) * 8, round(height / 8) * 8
        image_pil = image_pil.resize((width, height))
        image = np.array(image_pil)
        inpaint_mask_pil = inpaint_mask_pil.resize((width, height), Image.NEAREST)
        inpaint_mask = np.array(inpaint_mask_pil)

    # Common pipeline parameters
    common_params = {
        'image': image_pil,
        'mask_image': inpaint_mask_pil,
        'height': height,
        'width': width,
        'guidance_scale': 1.0,
        'num_inference_steps': 8,
        'strength': inpaint_strength,
        'output_type': 'np'
    }

    # Run pipeline
    if pipe_id == 'v1-5':
        inpainted = pipe(
            prompt_embeds=pipe.cached_prompt_embeds,
            **common_params
        ).images[0]
    else:
        inpainted = pipe(
            prompt_embeds=pipe.cached_prompt_embeds,
            pooled_prompt_embeds=pipe.cached_pooled_prompt_embeds,
            **common_params
        ).images[0]

    # Post-process results
    inpaint_mask = (inpaint_mask[..., np.newaxis] / 255).astype(np.uint8)
    # return (inpainted * 255).astype(np.uint8) * inpaint_mask + image * (1 - inpaint_mask)
    return (inpainted * 255).astype(np.uint8)