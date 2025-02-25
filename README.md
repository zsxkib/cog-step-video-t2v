# Step Video T2V Cog Implementation 🎥

[![Replicate](https://replicate.com/zsxkib/step-video-t2v/badge)](https://replicate.com/zsxkib/step-video-t2v)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Card-yellow)](https://huggingface.co/stepfun-ai/stepvideo-t2v-turbo)

Generate high-quality videos from text prompts using quantized models. Built with [Cog](https://cog.run) for seamless deployment.

## Quick Start

1. Install Cog following the [official guide](https://cog.run/docs/getting-started)
2. Run generation:
```bash
cog predict -i prompt="An astronaut dancing on the moon"
```

**Custom example:**
```bash
cog predict -i \
  prompt="A robotic cat chasing a laser pointer" \
  num_frames=75 \
  fps=30
```

## Features
- Text-to-video generation in seconds
- Optimized for GPU efficiency (~40GB VRAM)
- Built-in temporal denoising
- Easy deployment via Cog

## Try It Online
[![Run on Replicate](https://img.shields.io/badge/Replicate-Run%20Now-blue)](https://replicate.com/zsxkib/step-video-t2v)

## Community
Follow for updates and share your creations: [@zsakib_](https://x.com/zsakib_)

---

> Powered by [StepFun's StepVideo-T2V-Turbo](https://huggingface.co/stepfun-ai/stepvideo-t2v-turbo)
