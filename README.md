# Moxin-XD


This is a distilled LoRA based on [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), which enables efficient 4-step image generation. It can be used with the [diffusers](https://github.com/initml/diffusers) pipeline directly.

## Setup
To setting up and running, you need first to create a virtual env with at least python3.10 installed and activate it

Here is an example to create the venv environment using conda for cuda version 12.1.
```bash
conda create -n diffusion python=3.10
conda activate diffusion

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install accelerate transformers diffusers peft
```

## Inference with a diffusers pipeline

```python
from diffusers import DiffusionPipeline, LCMScheduler

# Load the base SDXL Pipeline
pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  use_safetensors=True,
)
# Scheduler
pipe.scheduler = LCMScheduler.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  subfolder="scheduler",
  timestep_spacing="trailing",
)
pipe.to("cuda")

# Load LoRA
pipe.load_lora_weights("path/to/lora/folder")
pipe.fuse_lora()

prompt = "A raccoon reading a book in a lush forest."

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
```
