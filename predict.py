# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import torch
from typing import List
from cog import BasePredictor, Input, Path
import subprocess
import time

from modelscope import snapshot_download
from diffsynth import ModelManager, StepVideoPipeline, save_video

###############################################################################
# You can set STEPVIDEO_MODE to one of: "standard", "quantized", or "low_vram"
# ENV example:
#   STEPVIDEO_MODE=quantized cog build
#   cog predict -i prompt="..."   # uses float8 for diffusion
#
# If STEPVIDEO_MODE is "low_vram", we still load bfloat16 but also default
# to more aggressive VRAM usage settings in setup below.
###############################################################################

MODEL_CACHE = "models"
BASE_URL = f"https://weights.replicate.delivery/default/StepVideo/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. "
            f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
    print("[!] Download took:", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Attempt custom tar model download first, else fallback
        try:
            model_files = ["stepfun-ai.tar"]
            if not os.path.exists(MODEL_CACHE):
                os.makedirs(MODEL_CACHE)

            for model_file in model_files:
                url = BASE_URL + model_file
                filename = url.split("/")[-1]
                dest_path = os.path.join(MODEL_CACHE, filename)
                if not os.path.exists(dest_path.replace(".tar", "")):
                    download_weights(url, dest_path)
        except Exception:
            print("[!] Custom tar download failed. Using Modelscope fallback.")
            snapshot_download(model_id="stepfun-ai/stepvideo-t2v", cache_dir=MODEL_CACHE)

        # Load compiled attention library
        lib_path = f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so"
        if os.path.exists(lib_path):
            try:
                torch.ops.load_library(lib_path)
                print("[setup] Successfully loaded compiled attention library.")
            except Exception as e:
                print("[setup] Could not load compiled attention library:", e)

        # Create model manager and load models
        self.model_manager = ModelManager()
        
        # 1) text encoder in float32
        self.model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
            torch_dtype=torch.float32,
            device="cpu"
        )
        
        # 2) step_llm and diffusion in bfloat16
        self.model_manager.load_models(
            [
                "models/stepfun-ai/stepvideo-t2v/step_llm",
                [
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
                ],
            ],
            torch_dtype=torch.bfloat16,
            device="cpu"
        )

        # 3) VAE in bfloat16
        self.model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors"],
            torch_dtype=torch.bfloat16,
            device="cpu"
        )

        # Create pipeline
        self.pipe = StepVideoPipeline.from_model_manager(
            self.model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

        # Enable VRAM management
        self.pipe.enable_vram_management(num_persistent_param_in_dit=0)

    def predict(
        self,
        prompt: str = Input(description="Prompt text", default="An astronaut on the Moon..."),
        negative_prompt: str = Input(description="Negative prompt", default="low resolution, text"),
        num_inference_steps: int = Input(description="Steps", default=30, ge=1, le=50),
        cfg_scale: float = Input(description="CFG scale", default=9.0),
        num_frames: int = Input(description="How many frames to create", default=51, ge=8),
        seed: int = Input(description="Random seed", default=42),
        tiled: bool = Input(description="Enable tiled generation", default=False),
        tile_size: int = Input(description="Tile size", default=34),
        tile_stride: int = Input(description="Tile stride", default=16),
    ) -> Path:
        """Run a single prediction on the model"""
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "num_frames": num_frames,
            "seed": seed,
            "tiled": tiled,
            "tile_size": (tile_size, tile_size),
            "tile_stride": (tile_stride, tile_stride),
        }

        print("[predict] Starting generation with:", pipeline_args)
        video = self.pipe(**pipeline_args)
        out_path = "video.mp4"
        
        # Convert video to correct format
        if isinstance(video, list):
            import numpy as np
            video = np.stack(video)
        
        # Ensure video is in the format [T, H, W, C]
        if video.shape[-1] != 3:  # If channels are not last
            video = np.moveaxis(video, 1, -1)  # Move channels to last dimension
        
        # Ensure values are in uint8 range [0, 255]
        if video.dtype != np.uint8:
            video = (video * 255).clip(0, 255).astype(np.uint8)
        
        # Ensure dimensions are divisible by 16
        h, w = video.shape[1:3]
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        
        if h != new_h or w != new_w:
            import torch.nn.functional as F
            # Temporarily move channels first for padding
            video = np.moveaxis(video, -1, 1)
            video = F.pad(torch.from_numpy(video), (0, new_w - w, 0, new_h - h), mode='replicate').numpy()
            # Move channels back to last position
            video = np.moveaxis(video, 1, -1)

        save_video(
            video, 
            out_path, 
            fps=25, 
            quality=5,
            ffmpeg_params=["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
        )

        print(f"[predict] Done! Saved video to {out_path}")
        return Path(out_path)
