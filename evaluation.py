from pathlib import Path
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import json
from time import perf_counter
from contextlib import contextmanager
from diffusers import AutoPipelineForInpainting, LCMScheduler, AutoencoderTiny
from utils.drag import bi_warp
from utils.evaluator import DragEvaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image processing with inpainting and evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing image folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    return parser.parse_args()

def get_preview_image(image, points, mask):
    """Generate preview image with warping effect"""
    handle_pts, target_pts, inpaint_mask = bi_warp(mask, points)
    image[target_pts[:, 1], target_pts[:, 0]] = image[handle_pts[:, 1], handle_pts[:, 0]]
    return image, inpaint_mask

@contextmanager
def timer(times_dict, key):
    """Context manager for timing code execution"""
    start = perf_counter()
    yield
    times_dict[key] = perf_counter() - start

class ImageProcessor:
    """Handles image processing operations including inpainting and evaluation"""
    
    def __init__(self, device='cuda'):
        """Initialize processor with specified device"""
        self.device = device
        self.pipe = self._setup_pipeline()
        self.evaluator = DragEvaluator()

    def _setup_pipeline(self):
        """Setup and configure the inpainting pipeline"""
        pipe = AutoPipelineForInpainting.from_pretrained(
            'runwayml/stable-diffusion-inpainting',
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        )
        
        # Configure pipeline components
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
        pipe.generator = torch.Generator(device=self.device).manual_seed(42)
        pipe = pipe.to(self.device)
        
        # Cache prompt embeddings for faster inference
        pipe.cached_prompt_embeds = pipe.encode_prompt(
            '', device=self.device, 
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )[0]
        
        return pipe

    @staticmethod
    def load_image_data(image_path, meta_path):
        """Load image and its metadata from files"""
        image = np.array(Image.open(image_path).convert('RGB'))
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        return image, data
    
    @staticmethod
    def load_points(pickle_path):
        """Load handle and target points from pickle file"""
        with open(pickle_path, 'rb') as f:
            points = pickle.load(f)['points']
            return np.array(points[0:-1:2]), np.array(points[1::2])

    def process_single_image(self, input_path, output_path):
        """Process a single image with timing for each step"""
        times = {}
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        with timer(times, 'load'):
            image, data = self.load_image_data(
                input_path / 'original_image.png',
                input_path / 'meta_data_i4p.pkl'
            )
            points, mask = data['points'], data['mask']

        # Generate preview and mask
        with timer(times, 'warp'):
            preview_img, inpaint_mask = get_preview_image(image.copy(), points, mask)
            preview_img_pil = Image.fromarray(preview_img)
            inpaint_mask_pil = Image.fromarray(inpaint_mask*255)
            ori_W, ori_H = inpaint_mask_pil.size
            
            # Resize images for model input
            preview_img_pil_resized = preview_img_pil.resize((512, 512))
            inpaint_mask_pil_resized = inpaint_mask_pil.resize((512, 512))

        # Perform inpainting
        with timer(times, 'inpaint'):
            inpainted = self.pipe(
                prompt_embeds=self.pipe.cached_prompt_embeds,
                image=preview_img_pil_resized, 
                mask_image=inpaint_mask_pil_resized,
                guidance_scale=1.0,
                num_inference_steps=4,
                strength=1.0,
                output_type='pil'
            ).images[0].resize((ori_W, ori_H))

            final_result = Image.fromarray((preview_img * (1 - mask[..., None]) + np.array(inpainted) * mask[..., None]).astype(np.uint8))

        # Save results
        with timer(times, 'save'):
            final_result.save(output_path / 'dragged_image.png')

        return times

    def evaluate_image(self, input_path, output_path, plot=False):
        """Evaluate processed image and compute metrics"""
        result_path = output_path / 'dragged_image.png'
        if not result_path.exists():
            return None

        original_img = np.array(Image.open(input_path / 'original_image.png'))
        edited_img = np.array(Image.open(result_path))
        handle_points, target_points = self.load_points(input_path / 'meta_data.pkl')

        plot_path = output_path / f"visualization.png" if plot else None
        
        return {
            'sample_name': input_path.name,
            'original_path': str(input_path),
            'lpips': self.evaluator.compute_lpips(original_img, edited_img),
            'distance': self.evaluator.compute_distance(
                original_img, edited_img,
                handle_points, target_points,
                plot_path=plot_path
            )
        }


def main():
    """Main execution function"""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all leaf directories (directories with no subdirectories)
    leaf_dirs = [d for d in data_dir.rglob("*") if d.is_dir() and not any(p.is_dir() for p in d.iterdir())]
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process images while maintaining directory structure
    all_times = []
    for image_dir in tqdm(leaf_dirs, desc="Processing images"):
        rel_path = image_dir.relative_to(data_dir)
        output_subdir = output_dir / rel_path
        times = processor.process_single_image(image_dir, output_subdir)
        all_times.append(times)

    # Clean up GPU memory
    del processor.pipe

    # Evaluate processed images
    metrics = []
    for image_dir in tqdm(leaf_dirs, desc="Evaluating results"):
        rel_path = image_dir.relative_to(data_dir)
        output_subdir = output_dir / rel_path
        result = processor.evaluate_image(image_dir, output_subdir, plot=args.plot)
        if result:
            metrics.append(result)

    # Calculate and save metrics
    avg_metrics = {
        'avg_lpips': np.mean([m['lpips'] for m in metrics]),
        'avg_distance': np.mean([m['distance'] for m in metrics])
    }

    results = {
        'individual_metrics': metrics,
        'average_metrics': avg_metrics,
        'processing_info': {
            'total_samples': len(leaf_dirs),
            'processed_samples': len(metrics),
            'data_dir': str(data_dir),
            'output_dir': str(output_dir)
        }
    }

    # Save results to JSON
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print statistics
    print("\nEvaluation Results:")
    print(f"Average LPIPS: {avg_metrics['avg_lpips']:.4f}")
    print(f"Average Distance: {avg_metrics['avg_distance']:.4f}")
    print(f"Total samples evaluated: {len(metrics)}")

    # Print timing statistics
    avg_times = {k: np.mean([t[k] for t in all_times]) for k in all_times[0].keys()}
    total_time = sum(sum(t.values()) for t in all_times)
    
    print("\nAverage times per image:")
    for step, time in avg_times.items():
        print(f"{step:8s}: {time:.3f}s")
    print(f"\nTotal: {total_time:.2f}s for {len(leaf_dirs)} images ({total_time/len(leaf_dirs):.2f}s per image)")

if __name__ == "__main__":
    main()