# Open‑Sora Runbook

This runbook summarizes how to install, configure and use the **Open‑Sora** text‑to‑video (T2V) and image‑to‑video (I2V) model.  Open‑Sora is a research project that aims to democratize high‑quality video generation by releasing an open‑source implementation and tools.  The project’s authors describe it as an initiative to **efficiently produce high‑quality video** while keeping the model, tools and documentation accessible to all【418668245440051†L61-L69】.  By embracing open‑source principles, Open‑Sora seeks to foster innovation, creativity and inclusivity in the field of video content generation【418668245440051†L61-L69】.

## 1. Prerequisites

* **Hardware:** A Linux machine with one or more NVIDIA GPUs is recommended for inference.  The model can run on a single GPU for 256×256 resolution, but higher resolutions (e.g., 768×768) benefit from multiple GPUs using tensor parallelism【418668245440051†L154-L167】.
* **Python environment:** Use Python 3.10 and install dependencies in an isolated environment such as conda or venv.
* **CUDA and drivers:** Ensure the CUDA toolkit and drivers are compatible with PyTorch ≥ 2.4.0.

## 2. Installation

1. **Create an environment.**  The Quickstart instructions suggest using conda:

   ```bash
   conda create -n opensora python=3.10
   conda activate opensora
   ```

2. **Clone the source code.**  Download the Open‑Sora repository from GitHub:

   ```bash
   git clone https://github.com/hpcaitech/Open-Sora
   cd Open-Sora
   ```

3. **Install the package.**  Install Open‑Sora in development mode and install key dependencies.  The project requires PyTorch ≥ 2.4.0; you may install the package and optional acceleration libraries as follows【418668245440051†L95-L119】:

   ```bash
   pip install -v .  # or: pip install -v -e . for editable mode
   # install xformers (choose the correct CUDA version)
   pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
   # install flash attention (faster attention kernels)
   pip install flash-attn --no-build-isolation
   # optional: Flash Attention 3 for additional speed (from source)
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention/hopper
   python setup.py install
   ```

## 3. Downloading Model Weights

Open‑Sora 2.0 provides an 11 billion parameter model supporting both **T2V** and **I2V** tasks at 256 px and 768 px resolutions【418668245440051†L121-L124】.  Download the checkpoints to `./ckpts` using one of the following methods【418668245440051†L120-L134】:

* **Hugging Face**:  
  
  ```bash
  pip install "huggingface_hub[cli]"
  huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
  ```

* **ModelScope**:  
  
  ```bash
  pip install modelscope
  modelscope download hpcai-tech/Open-Sora-v2 --local_dir ./ckpts
  ```

After downloading, ensure the paths in your inference configuration point to the checkpoint directory.

## 4. Text‑to‑Video Generation

Open‑Sora is optimized for image‑to‑video generation but can produce videos directly from text prompts.  The project uses a text‑to‑image‑to‑video pipeline for high quality results【418668245440051†L135-L139】.  The following commands demonstrate how to generate videos from text using `torchrun` (PyTorch’s distributed launcher).  Replace the prompt with your own description and set `--save-dir` to the output folder.

### 4.1 Generate 256×256 resolution videos

```bash
# Single GPU, direct text→image→video pipeline
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_256px.py \
    --save-dir samples \
    --prompt "A rainy scene by the sea"

# Generate multiple videos from a CSV of prompts (one prompt per row)
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_256px.py \
    --save-dir samples \
    --dataset.data-path assets/texts/example.csv
```

### 4.2 Generate 768×768 resolution videos

Larger videos require more memory.  You can run on one GPU or distribute across eight GPUs using ColossalAI’s sequence parallelism【418668245440051†L154-L162】:

```bash
# Single GPU
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_768px.py \
    --save-dir samples \
    --prompt "A rainy scene by the sea"

# Multi‑GPU (8 GPUs)
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_768px.py \
    --save-dir samples \
    --prompt "A rainy scene by the sea"
```

**Parameters:**  Use `--aspect_ratio` to control the video’s aspect ratio (values like `16:9`, `9:16`, `1:1`, `2.39:1`) and `--num_frames` to control the length.  The number of frames should follow the pattern `4k+1` and be less than 129【418668245440051†L164-L167】.

## 5. Image‑to‑Video Generation

Open‑Sora also supports conditioning on an image.  Provide a reference image and a prompt to generate a video【418668245440051†L178-L201】.  For example, to create a 256 px video conditioned on an image stored at `assets/texts/i2v.png`:

```bash
# 256 px single GPU
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --cond_type i2v_head \
    --prompt "A plump pig wallows in a muddy pond on a rustic farm..." \
    --ref assets/texts/i2v.png \
    --save-dir samples

# 768 px multi‑GPU
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/768px.py \
    --cond_type i2v_head \
    --dataset.data-path assets/texts/i2v.csv \
    --save-dir samples
```

The CSV file should contain paths to reference images and corresponding prompts.  The configuration files control hyper‑parameters; adjust them as needed.

## 6. Advanced Features

### 6.1 Motion Score

During training, Open‑Sora injects a **motion score** into the prompt.  At inference time, you can specify a fixed motion score or ask the model to compute one dynamically【418668245440051†L203-L221】:

```bash
# Generate a video with motion score 4
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_256px.py \
    --save-dir samples --prompt "A rainy scene by the sea" --motion-score 4

# Dynamically compute motion score (requires OpenAI API key)
export OPENAI_API_KEY=sk-...  # set your key
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_256px.py \
    --save-dir samples --prompt "A rainy scene by the sea" --motion-score dynamic
```

### 6.2 Prompt Refinement

Open‑Sora integrates ChatGPT to refine prompts.  To refine a prompt during inference, set the `--refine-prompt` flag【418668245440051†L225-L233】:

```bash
export OPENAI_API_KEY=sk-...  # set your key
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_256px.py \
    --save-dir samples --prompt "A rainy scene by the sea" --refine-prompt True
```

### 6.3 Reproducibility

To make results reproducible, set seeds via `--sampling_option.seed` and `--seed`【418668245440051†L238-L244】.  Use `--num-sample k` to generate multiple samples per prompt.

## 7. Troubleshooting

* **CUDA out of memory:** Reduce batch size, lower resolution, or use multi‑GPU parallelism.
* **Invalid shape or device mismatch:** Ensure your environment uses the correct CUDA version and that `xformers` and `flash‑attn` are installed for your GPU.
* **Slow inference:** Install Flash Attention 3 and use half‑precision (`--dtype fp16`) to speed up inference.

## 8. References

Open‑Sora is developed by the PKU YuanGroup and collaborators.  For more details, refer to the official paper and technical report【418668245440051†L61-L69】【418668245440051†L90-L93】.  The project draws on technologies such as **ColossalAI**, **DiT**, **Flash Attention**, **PixArt**, **Flux** and many others【418668245440051†L246-L299】.
