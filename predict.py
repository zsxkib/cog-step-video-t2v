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
        """Load the model into memory exactly once for faster re-use."""
        # Ensure model cache directory exists
        os.makedirs(MODEL_CACHE, exist_ok=True)
        
        # 1) Download or ensure model is present
        try:
            # Attempt custom tar download
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

        # 2) Optionally load compiled attention kernel
        lib_path = f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so"
        if os.path.exists(lib_path):
            try:
                torch.ops.load_library(lib_path)
                print("[setup] Successfully loaded compiled attention library")
            except Exception as e:
                print("[setup] Could not load compiled attention library:", e)
        
        # 3) Build pipeline once and store in self.pipe
        print("[setup] Building pipeline for inference...")
        self.model_manager = ModelManager()

        # a) Text encoder, float32 on CPU
        self.model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
            torch_dtype=torch.float32,
            device="cpu"
        )

        # b) Step LLM & diffusion, float8 on CPU
        diffusion_assets = [
            "models/stepfun-ai/stepvideo-t2v/step_llm",
            [
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
                "models/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
        ]
        self.model_manager.load_models(
            diffusion_assets,
            torch_dtype=torch.float8_e4m3fn,
            device="cpu"
        )

        # c) VAE, bfloat16 on CPU
        self.model_manager.load_models(
            ["models/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors"],
            torch_dtype=torch.bfloat16,
            device="cpu"
        )

        # d) Create pipeline on GPU with bfloat16
        self.pipe = StepVideoPipeline.from_model_manager(
            self.model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

        # e) Enable VRAM management
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        print("[setup] Pipeline is ready.")

    def predict(
        self,
        prompt: str = Input(description="Prompt text", default="An astronaut on the moon"),
        negative_prompt: str = Input(description="Negative prompt", default="low resolution, text"),
        num_inference_steps: int = Input(description="Number of inference steps", default=30),
        cfg_scale: float = Input(description="Classifier free guidance scale", default=9.0),
        num_frames: int = Input(description="Number of frames", default=51),
        fps: int = Input(description="Frames per second in output video", default=25),
        quality: int = Input(description="Video quality (0-10, 10 is highest quality)", default=5),
        seed: int = Input(description="Random seed. Leave blank for random", default=None),
    ) -> Path:
        """Run inference using the pre-loaded pipeline."""
        # 1) Handle random seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"[predict] Using random seed: {seed}")
        torch.manual_seed(seed)

        # 2) Run the pipeline
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            num_frames=num_frames,
            seed=seed
        )

        # 3) Save and return the video
        out_path = "video.mp4"
        
        # Apply temporal denoising to reduce flickering
        ffmpeg_params = ["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
        
        # Save video
        save_video(
            video,
            out_path,
            fps=fps,
            quality=quality,
            ffmpeg_params=ffmpeg_params
        )
        print(f"[predict] Video saved to {out_path}")
        
        return Path(out_path)
