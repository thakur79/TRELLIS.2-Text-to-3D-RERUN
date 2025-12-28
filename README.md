# **TRELLIS.2-Text-to-3D-RERUN**

> A Gradio with Rerun Embedded demonstration for Microsoft's TRELLIS.2-4B model with integrated Rerun visualization. Converts text prompts or uploaded images into high-quality textured 3D assets (GLB) via a two-stage workflow: Text-to-Image (Z-Image-Turbo) → Image-to-3D (TRELLIS.2). Features interactive 3D viewing powered by Rerun SDK, with proper coordinate system setup, axes helpers, and downloadable GLB files.

## Features

- **Text-to-Image-to-3D**: Generate base images from prompts using Z-Image-Turbo, then lift to 3D.
- **Direct Image-to-3D**: Upload RGBA/PNG images; auto-preprocesses with background removal (BRIA-RMBG-2.0) and cropping.
- **Rerun 3D Viewer**: Interactive visualization with correct RIGHT_HAND_Y_UP coordinates, colored axes (X=red, Y=green, Z=blue), and clean 3D view blueprint.
- **Advanced Controls**: Resolutions (512/1024/1536), detailed sampler settings for sparse structure, shape, and material stages, face decimation, texture size.
- **Robust Export**: GLB with PNG textures (extension_webp=False for compatibility); fallback remeshing if high-quality fails.
- **Session Management**: Per-user temp directories; auto-cleanup on unload.
- **Custom Theme**: OrangeRedTheme with responsive layout.
- **Rich Examples**: 70+ image inputs and 60+ text prompts (cats, planes, cars, furniture, etc.).

---

<img width="1207" height="750" alt="Screenshot 2025-12-28 at 23-41-16 TRELLIS 2-Text-to-3D - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/7c0d7eaf-36c3-4e52-9fbb-fb09e370cbce" />

<img width="1207" height="750" alt="Screenshot 2025-12-28 at 23-37-13 TRELLIS 2-Text-to-3D - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/97fd118e-aa96-480a-a05a-f4ba29dd3dea" />

---

https://github.com/user-attachments/assets/4e34bce8-7dc3-4cdc-a413-b170eb9eba93

https://github.com/user-attachments/assets/a335d0af-21b2-4826-af81-902f2fafce9c

---

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (required for bfloat16 and optimizations).
- pip >= 23.0.0 (see pre-requirements.txt).
- Stable internet for initial model downloads.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-RERUN.git
   cd TRELLIS.2-Text-to-3D-RERUN
   ```

2. Install pre-requirements:
   Create a `pre-requirements.txt` file with the following content, then run:
   ```
   pip install -r pre-requirements.txt
   ```

   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```

3. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   --extra-index-url https://download.pytorch.org/whl/cu124
   git+https://github.com/huggingface/diffusers.git@refs/pull/12790/head
   torch==2.6.0
   torchvision==0.21.0
   triton==3.2.0
   pillow==12.0.0
   matplotlib
   rembg
   imageio==2.37.2
   imageio-ffmpeg==0.6.0
   tqdm==4.67.1
   easydict==1.13
   opencv-python-headless==4.12.0.88
   trimesh==4.10.1
   zstandard==0.25.0
   kornia==0.8.2
   timm==1.0.22
   git+https://github.com/huggingface/transformers.git@v4.57.3
   git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl
   omegaconf
   termcolor
   icecream
   pyserde
   gradio
   rerun-sdk
   gradio_rerun
   scipy
   jax
   jaxtyping
   monopriors
   braceexpand
   ```

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage

1. **Text-to-Image-to-3D**:
   - Enter prompt (e.g., "A cyberpunk Cat 3D").
   - Click "1.Generate Image".
   - Proceed to 3D.

2. **Image-to-3D**:
   - Upload image directly.

3. **Configure**:
   - Resolution, sampler params, faces/texture size.

4. **Generate 3D**: Click "2.Generate 3D".

5. **Output**:
   - Interactive Rerun viewer with proper 3D orientation.
   - Download GLB button.

## Rerun Viewer

- Correct coordinate system (RIGHT_HAND_Y_UP).
- Axes helpers for orientation.
- Clean blueprint view.
- Recordings saved in `tmp/` as `.rrd`.

## Troubleshooting

- **Rerun Issues**: Ensure `gradio_rerun` and `rerun-sdk`; blueprint optional.
- **Export Fails**: Fallback remesh=False; aggressive simplification to 1M faces.
- **OOM**: Reduce resolution/steps; clear cache.
- **Preprocessing**: BRIA-RMBG requires internet.

Repository: [https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-RERUN.git](https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-RERUN.git)

## Contributing

Contributions welcome! Enhance Rerun blueprints, add examples, or optimize post-processing.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
