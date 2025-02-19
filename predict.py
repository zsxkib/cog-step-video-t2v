# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import torch
from typing import List
from cog import BasePredictor, Input, Path

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
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Try custom download first
        try:
            # Original download code unchanged
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
            # 1. Download the model from Modelscope
            print("Custom download failed, using Modelscope fallback")
            snapshot_download(model_id="stepfun-ai/stepvideo-t2v", cache_dir=MODEL_CACHE)

        # 2. Optionally load the compiled attention ops for the LLM text encoder
        lib_path = f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so"
        if os.path.exists(lib_path):
            torch.ops.load_library(lib_path)

        # 3. Create a ModelManager and load models
        self.model_manager = ModelManager()

        # Load text encoder on CPU
        self.model_manager.load_models(
            [f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin"],
            torch_dtype=torch.float32,
            device="cpu"
        )

        # Load LLM/diffusion & VAE on GPU in bfloat16
        self.model_manager.load_models(
            [
                f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/step_llm",
                [
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
                    f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
                ]
            ],
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.model_manager.load_models(
            [f"{MODEL_CACHE}/stepfun-ai/stepvideo-t2v/vae/vae_v2.safetensors"],
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

        # 4. Create the pipeline
        self.pipe = StepVideoPipeline.from_model_manager(
            self.model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )

        # 5. Optional VRAM management. On an H100, likely not needed:
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None)

    def predict(
        self,
        prompt: str = Input(
            description="Descriptive text for what you want to see in the video.",
            default="An astronaut on the Moon discovers a monolith with the text 'STEPFUN' glowing brightly. Ultra HD, HDR video, highly detailed."
        ),
        negative_prompt: str = Input(
            description="Negative prompt for undesirable artifacts.",
            default="dark, low-resolution, extra hands, watermark, text overlays"
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps.", default=30
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale.", default=9.0
        ),
        num_frames: int = Input(
            description="Number of frames in the generated video.", default=51
        ),
        seed: int = Input(
            description="Seed for reproducibility.", default=42
        ),
        tiled: bool = Input(
            description="Enable memory-efficient tiled generation",
            default=False
        ),
        tile_size: int = Input(
            description="Size of each tile (applies to both height and width)",
            default=34
        ),
        tile_stride: int = Input(
            description="Stride between tiles (applies to both directions)",
            default=16
        ),
    ) -> Path:

        # Prepare pipeline arguments
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "num_frames": num_frames,
            "seed": seed,
        }

        # Enable tile-based generation if requested
        if tiled:
            pipeline_args.update({
                "tiled": True,
                "tile_size": (tile_size, tile_size),
                "tile_stride": (tile_stride, tile_stride)
            })

        # Run the pipeline
        video = self.pipe(**pipeline_args)

        # Save video result to disk
        out_path = "output.mp4"
        save_video(
            video, out_path, fps=25, quality=5,
            ffmpeg_params=["-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1"]
        )
        return Path(out_path)
