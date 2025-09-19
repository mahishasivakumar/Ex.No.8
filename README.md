## Exp 8: Reproducing an Image Using Prompts for Image Generation

# Nmae : Mahisha S
# Reg. No: 212222040095

## Aim

   To reproduce a given image using text-to-image generation models through prompt engineering and iterative refinement.

   Apparatus / Tools

   Software: Python 3.9+, Stable Diffusion / DALL·E / Midjourney

   Libraries: diffusers, transformers, torch, PIL
  
   Hardware: GPU-enabled machine (NVIDIA preferred)

   Dataset: Target image(s) stored in target_images/

## Theory

Modern text-to-image models generate images from textual descriptions.
The reproduction process involves:

Observing the target image

Extracting descriptive features (subject, style, lighting, composition)

Converting them into prompt templates

Iteratively refining prompts until the generated output visually resembles the target.

Metrics like SSIM (Structural Similarity) or CLIP similarity can be used to measure closeness.

## Procedure

Place a target image in target_images/img01.jpg.

Analyze its features (e.g., subject: cat sitting on a window sill; lighting: natural daylight; style: realistic photograph).

Write initial prompts. Example:

A cat sitting on a window sill, realistic photograph.

A close-up photo of a cat sitting on a window sill, natural daylight, realistic, detailed texture.

Generate images using Stable Diffusion or chosen tool.

Refine prompts by adding/removing attributes.

Save outputs and compare them with the target.

Record observations and compute similarity metrics.

Structured Code Example (Stable Diffusion with Diffusers)
from diffusers import StableDiffusionPipeline
import torch, os
from PIL import Image

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Target image
target = Image.open("target_images/img01.jpg").convert("RGB")

# Prompts
prompts = [
    "A cat sitting on a window sill, realistic photograph.",
    "A close-up photo of a cat sitting on a window sill, natural daylight, realistic, detailed texture."
]

output_dir = "outputs/exp8"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for i, prompt in enumerate(prompts):
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=40).images[0]
    image.save(os.path.join(output_dir, f"gen_{i+1}.png"))
    print(f"Saved: gen_{i+1}.png")

# Sample Images (Illustrative)

# Target Image:

Generated Output 1 (simple prompt):

Generated Output 2 (refined prompt):

(Note: The above are sample placeholders; replace with your generated images in outputs/exp8/)

## Observations
Prompt Version	Features Captured	Similarity (Visual)	Remarks
Simple Prompt	Subject only	Moderate	Background & lighting missing
Refined Prompt	Subject + Lighting + Style	High	Closely resembles target
Result

The refined prompt reproduced the target image more accurately, demonstrating that prompt specificity improves fidelity.

## Result:

The experiment successfully explored how structured, conditional, and creative prompting affects the quality of AI-generated audio. Rich, specific prompts and thoughtful refinement enhanced control over the output. By leveraging the strengths of tools like Jukebox, AudioLM, and MusicGen, users can generate high-quality audio for music, speech, and soundscapes tailored to a wide range of applications.

