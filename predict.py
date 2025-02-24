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
# This script can dynamically switch between the following modes in a single
# container run:
#   • "standard"  (bfloat16 diffusion, ~80G VRAM)
#   • "quantized" (float8 diffusion, ~40G VRAM)
#   • "low_vram"  (bfloat16 diffusion but aggressively reduces VRAM usage)
#
# Some parameters, such as 'tiled', are only intended for low-level VRAM usage.
#   • standard/quantized modes do not usually benefit from 'tiled' 
#     (though they won't error if you manually set it).
#   • low_vram mode sets 'tiled' to True by default to reduce GPU usage.
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
        """Load the model into memory to be used repeatedly for inference"""
        # Initialize mode configurations
        self.pipelines = {}
        self.mode_config = {
            "standard": {
                "diffusion_dtype": torch.bfloat16,
                "vram_params": None,      # Unlimited VRAM (~80G)
                "default_tiled": False,
                "default_steps": 30,
                "load_vae_with_diffusion": True,  # Standard loads VAE with diffusion
            },
            "quantized": {
                "diffusion_dtype": torch.float8_e4m3fn,
                "vram_params": None,      # Unlimited VRAM (~40G)
                "default_tiled": False,
                "default_steps": 30,
                "load_vae_with_diffusion": False,  # Load VAE separately
            },
            "low_vram": {
                "diffusion_dtype": torch.bfloat16,
                "vram_params": 0,         # Aggressive VRAM management (~24G)
                "default_tiled": True,    # Enable tiling by default
                "default_steps": 2,       # Fewer steps for low VRAM mode
                "load_vae_with_diffusion": False,  # Load VAE separately
            }
        }

        # Download models if needed
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
                print("[setup] Successfully loaded compiled attention library")
            except Exception as e:
                print("[setup] Could not load compiled attention library:", e)

    def _load_pipeline_for_mode(self, mode: str) -> StepVideoPipeline:
        """Load pipeline for specified mode if not already loaded"""
        if mode in self.pipelines:
            return self.pipelines[mode]

        config = self.mode_config[mode]
        print(f"[load_pipeline] Loading '{mode}' pipeline...")
        
        # Force single GPU usage
        device = "cuda:0"
        model_manager = ModelManager()
        
        # 1. Text encoder (always float32)
        model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
            torch_dtype=torch.float32,
            device=device  # Changed from "cpu" to device
        )

        # 2. Step LLM and diffusion (with optional VAE for standard mode)
        diffusion_paths = [
            "models/stepfun-ai/stepvideo-t2v/step_llm",
        ]
        
        if config["load_vae_with_diffusion"]:
            diffusion_paths.append("models/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors")
            
        diffusion_paths.append([
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
        ])
        
        model_manager.load_models(
            diffusion_paths,
            torch_dtype=config["diffusion_dtype"],
            device=device  # Changed from "cpu" to device
        )

        # 3. Load VAE separately for quantized/low_vram modes
        if not config["load_vae_with_diffusion"]:
            model_manager.load_models(
                ["models/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors"],
                torch_dtype=torch.bfloat16,
                device=device  # Changed from "cpu" to device
            )

        # Create pipeline
        pipe = StepVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch.bfloat16,
            device=device  # Changed from "cuda" to explicit "cuda:0"
        )
        
        # Configure VRAM management
        if config["vram_params"] is not None:
            pipe.enable_vram_management(num_persistent_param_in_dit=config["vram_params"])

        self.pipelines[mode] = pipe
        return pipe

    def predict(
        self,
        prompt: str = Input(description="Prompt text", default="An astronaut on the Moon"),
        negative_prompt: str = Input(description="Negative prompt", default="low resolution, text"),
        mode: str = Input(
            description="Mode: Choose between standard quantization, low VRAM optimizations, or both combined",
            default="low_vram",
            choices=["quantized", "low_vram", "quantized_low_vram"]
        ),
        num_inference_steps: int = Input(description="Number of inference steps", default=None),
        num_frames: int = Input(description="Number of frames", default=51),
        seed: int = Input(description="Random seed. Leave blank for random", default=None),
        cfg_scale: float = Input(description="Classifier free guidance scale", default=9.0),
    ) -> Path:
        """Run prediction"""
        use_low_vram = (mode == "low_vram" or mode == "quantized_low_vram")
        use_quantized = (mode == "quantized" or mode == "quantized_low_vram")
        
        if num_inference_steps is None:
            num_inference_steps = 2 if use_low_vram else 30

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"Using random seed: {seed}")

        # Load models exactly as in the examples
        model_manager = ModelManager()
        
        # 1. Text encoder (same for both modes)
        model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
            torch_dtype=torch.float32, 
            device="cpu"
        )

        # 2. Step LLM and diffusion
        diffusion_paths = [
            "models/stepfun-ai/stepvideo-t2v/step_llm",
            [
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
            ]
        ]
        
        # Use different dtype based on mode
        model_manager.load_models(
            diffusion_paths,
            torch_dtype=torch.float8_e4m3fn if use_quantized else torch.bfloat16,
            device="cpu"
        )

        # 3. VAE (same for both modes)
        model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors"],
            torch_dtype=torch.bfloat16,
            device="cpu"
        )

        # Create pipeline
        pipe = StepVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

        # Set VRAM management based on mode
        pipe.enable_vram_management(
            num_persistent_param_in_dit=0 if use_low_vram else None
        )

        # Run prediction with mode-specific settings
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            num_frames=num_frames,
            seed=seed,
            tiled=use_low_vram,
            tile_size=(34, 34) if use_low_vram else None,
            tile_stride=(16, 16) if use_low_vram else None,
        )

        # Save video
        out_path = "video.mp4"
        save_video(
            video,
            out_path,
            fps=25,
            quality=5,
            ffmpeg_params=["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
        )

        return Path(out_path)
