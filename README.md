<div align="center">

# SD.Next

<<<<<<< HEAD
**Stable Diffusion implementation with advanced features**

[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/vladmandic)
![Last Commit](https://img.shields.io/github/last-commit/vladmandic/automatic?svg=true)
![License](https://img.shields.io/github/license/vladmandic/automatic?svg=true)
[![Discord](https://img.shields.io/discord/1101998836328697867?logo=Discord&svg=true)](https://discord.gg/VjvR2tabEX)
=======
## Features
[Detailed feature showcase with images](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Color Sketch
- Prompt Matrix
- Stable Diffusion Upscale
- Attention, specify parts of text that the model should pay more attention to
    - a man in a `((tuxedo))` - will pay more attention to tuxedo
    - a man in a `(tuxedo:1.21)` - alternative syntax
    - select text and press `Ctrl+Up` or `Ctrl+Down` (or `Command+Up` or `Command+Down` if you're on a MacOS) to automatically adjust attention to selected text (code contributed by anonymous user)
- Loopback, run img2img processing multiple times
- X/Y/Z plot, a way to draw a 3 dimensional plot of images with different parameters
- Textual Inversion
    - have as many embeddings as you want and use any names you like for them
    - use multiple embeddings with different numbers of vectors per token
    - works with half precision floating point numbers
    - train embeddings on 8GB (also reports of 6GB working)
- Extras tab with:
    - GFPGAN, neural network that fixes faces
    - CodeFormer, face restoration tool as an alternative to GFPGAN
    - RealESRGAN, neural network upscaler
    - ESRGAN, neural network upscaler with a lot of third party models
    - SwinIR and Swin2SR ([see here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/2092)), neural network upscalers
    - LDSR, Latent diffusion super resolution upscaling
- Resizing aspect ratio options
- Sampling method selection
    - Adjust sampler eta values (noise multiplier)
    - More advanced noise setting options
- Interrupt processing at any time
- 4GB video card support (also reports of 2GB working)
- Correct seeds for batches
- Live prompt token length validation
- Generation parameters
     - parameters you used to generate images are saved with that image
     - in PNG chunks for PNG, in EXIF for JPEG
     - can drag the image to PNG info tab to restore generation parameters and automatically copy them into UI
     - can be disabled in settings
     - drag and drop an image/text-parameters to promptbox
- Read Generation Parameters Button, loads parameters in promptbox to UI
- Settings page
- Running arbitrary python code from UI (must run with `--allow-code` to enable)
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Tiling support, a checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
    - Can use a separate neural network to produce previews with almost none VRAM or compute requirement
- Negative prompt, an extra text field that allows you to list what you don't want to see in generated image
- Styles, a way to save part of prompt and easily apply them via dropdown later
- Variations, a way to generate same image but with tiny differences
- Seed resizing, a way to generate same image but at slightly different resolution
- CLIP interrogator, a button that tries to guess prompt from an image
- Prompt Editing, a way to change prompt mid-generation, say to start making a watermelon and switch to anime girl midway
- Batch Processing, process a group of files using img2img
- Img2img Alternative, reverse Euler method of cross attention control
- Highres Fix, a convenience option to produce high resolution pictures in one click without usual distortions
- Reloading checkpoints on the fly
- Checkpoint Merger, a tab that allows you to merge up to 3 checkpoints into one
- [Custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) with many extensions from community
- [Composable-Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), a way to use multiple prompts at once
     - separate prompts using uppercase `AND`
     - also supports weights for prompts: `a cat :1.2 AND a dog AND a penguin :2.2`
- No token limit for prompts (original stable diffusion lets you use up to 75 tokens)
- DeepDanbooru integration, creates danbooru style tags for anime prompts
- [xformers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), major speed increase for select cards: (add `--xformers` to commandline args)
- via extension: [History tab](https://github.com/yfszzx/stable-diffusion-webui-images-browser): view, direct and delete images conveniently within the UI
- Generate forever option
- Training tab
     - hypernetworks and embeddings options
     - Preprocessing images: cropping, mirroring, autotagging using BLIP or deepdanbooru (for anime)
- Clip skip
- Hypernetworks
- Loras (same as Hypernetworks but more pretty)
- A separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt 
- Can select to load a different VAE from settings screen
- Estimated completion time in progress bar
- API
- Support for dedicated [inpainting model](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion) by RunwayML
- via extension: [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients), a way to generate images with a specific aesthetic by using clip images embeds (implementation of [https://github.com/vicgalle/stable-diffusion-aesthetic-gradients](https://github.com/vicgalle/stable-diffusion-aesthetic-gradients))
- [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20) for instructions
- [Alt-Diffusion](https://arxiv.org/abs/2211.06679) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alt-diffusion) for instructions
- Now without any bad letters!
- Load checkpoints in safetensors format
- Eased resolution restriction: generated image's dimensions must be a multiple of 8 rather than 64
- Now with a license!
- Reorder elements in the UI from settings screen
- [Segmind Stable Diffusion](https://huggingface.co/segmind/SSD-1B) support

## Installation and Running
Make sure the required [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) are met and follow the instructions available for:
- [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended)
- [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.
- [Intel CPUs, Intel GPUs (both integrated and discrete)](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

[Wiki](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.gg/VjvR2tabEX) | [Changelog](CHANGELOG.md)

</div>
</br>

<<<<<<< HEAD
## Notable features

All individual features are not listed here, instead check [ChangeLog](CHANGELOG.md) for full list of changes
- Multiple backends!  
  ▹ **Original | Diffusers**
- Multiple diffusion models!  
  ▹ **Stable Diffusion 1.5/2.1 | SD-XL | LCM | Segmind | Kandinsky | Pixart-α | Würstchen | aMUSEd | DeepFloyd IF | UniDiffusion | SD-Distilled | BLiP Diffusion | etc.**
- Built-in Control for Text, Image, Batch and video processing!  
  ▹ **ControlNet | ControlNet XS | Control LLLite | T2I Adapters | IP Adapters**  
- Multiplatform!  
 ▹ **Windows | Linux | MacOS with CPU | nVidia | AMD | IntelArc | DirectML | OpenVINO | ONNX+Olive**
- Platform specific autodetection and tuning performed on install
- Optimized processing with latest `torch` developments with built-in support for `torch.compile` and multiple compile backends
- Improved prompt parser  
- Enhanced *Lora*/*LoCon*/*Lyco* code supporting latest trends in training  
- Built-in queue management  
- Enterprise level logging and hardened API  
- Modern localization and hints engine  
- Broad compatibility with existing extensions ecosystem and new extensions manager  
- Built in installer with automatic updates and dependency management  
- Modernized UI with theme support and number of built-in themes *(dark and light)*  
=======
### Installation on Windows 10/11 with NVidia-GPUs using release package
1. Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2. Run `update.bat`.
3. Run `run.bat`.
> For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

### Automatic Installation on Windows
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Newer version of Python does not support torch), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
# Red Hat-based:
sudo dnf install wget git python3 gperftools-libs libglvnd-glx 
# openSUSE-based:
sudo zypper install wget git python3 libtcmalloc4 libglvnd
# Arch-based:
sudo pacman -S wget git python3
```
2. Navigate to the directory you would like the webui to be installed and execute the following command:
```bash
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
```
3. Run `webui.sh`.
4. Check `webui-user.sh` for options.
### Installation on Apple Silicon
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

<br>

![Screenshot-Dark](html/xmas-default.jpg)
![Screenshot-Control](html/xmas-control.jpg)
![Screenshot-Light](html/light-teal.jpg)

<<<<<<< HEAD
<br>

## Backend support

**SD.Next** supports two main backends: *Original* and *Diffusers*:

- **Original**: Based on [LDM](https://github.com/Stability-AI/stablediffusion) reference implementation and significantly expanded on by [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  This is the default backend and it is fully compatible with all existing functionality and extensions  
  Supports **SD 1.x** and **SD 2.x** models  
  All other model types such as *SD-XL, LCM, PixArt, Segmind, Kandinsky, etc.* require backend **Diffusers**  
- **Diffusers**: Based on new [Huggingface Diffusers](https://huggingface.co/docs/diffusers/index) implementation  
  Supports *original* SD models as well as *all* models listed below  
  See [wiki article](https://github.com/vladmandic/automatic/wiki/Diffusers) for more information  

## Model support

Additional models will be added as they become available and there is public interest in them

- [RunwayML Stable Diffusion](https://github.com/Stability-AI/stablediffusion/) 1.x and 2.x *(all variants)*  
- [StabilityAI Stable Diffusion XL](https://github.com/Stability-AI/generative-models)  
- [StabilityAI Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) Base and XT  
- [LCM: Latent Consistency Models](https://github.com/openai/consistency_models)  
- [aMUSEd 256](https://huggingface.co/amused/amused-256) 256 and 512
- [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega)  
- [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B)  
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) *2.1 and 2.2 and latest 3.0*  
- [PixArt-α XL 2](https://github.com/PixArt-alpha/PixArt-alpha) *Medium and Large*  
- [Warp Wuerstchen](https://huggingface.co/blog/wuertschen)  
- [Playground](https://huggingface.co/playgroundai/playground-v2-256px-base) *v1, v2 256, v2 512, v2 1024*  
- [Tsinghua UniDiffusion](https://github.com/thu-ml/unidiffuser)
- [DeepFloyd IF](https://github.com/deep-floyd/IF) *Medium and Large*
- [ModelScope T2V](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
- [Segmind SD Distilled](https://huggingface.co/blog/sd_distillation) *(all variants)*
- [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/)  


Also supported are modifiers such as:
- **LCM** and **Turbo** (Adversarial Diffusion Distillation) networks
- All **LoRA** types such as LoCon, LyCORIS, HADA, IA3, Lokr, OFT
- **AnimateDiff** for SD 1.5
- **IP-Adapters** for SD 1.5 and SD-XL

> [!IMPORTANT]
> - Loading any model other than standard SD 1.x / SD 2.x requires use of backend **Diffusers**  
> - Loading any other models using **Original** backend is not supported  
> - Loading manually download model `.safetensors` files is supported for SD 1.x / SD 2.x / SD-XL models only  
> - For all other model types, use backend **Diffusers** and use built in Model downloader or  
  select model from Networks -> Models -> Reference list in which case it will be auto-downloaded and loaded  

## Platform support

- *nVidia* GPUs using **CUDA** libraries on both *Windows and Linux*  
- *AMD* GPUs using **ROCm** libraries on *Linux*  
  Support will be extended to *Windows* once AMD releases ROCm for Windows  
- *Intel Arc* GPUs using **OneAPI** with *IPEX XPU* libraries on both *Windows and Linux*  
- Any GPU compatible with *DirectX* on *Windows* using **DirectML** libraries  
  This includes support for AMD GPUs that are not supported by native ROCm libraries  
- Any GPU or device compatible with **OpenVINO** libraries on both *Windows and Linux*  
- *Apple M1/M2* on *OSX* using built-in support in Torch with **MPS** optimizations  
- *ONNX/Olive* (experimental)  

## Install

- [Step-by-step install guide](https://github.com/vladmandic/automatic/wiki/Installation)
- [Advanced install notes](https://github.com/vladmandic/automatic/wiki/Advanced-Install)
- [Common installation errors](https://github.com/vladmandic/automatic/discussions/1627)  
- [FAQ](https://github.com/vladmandic/automatic/discussions/1011)
- If you can't run us locally, try our friends at [RunDuffusion!](https://rundiffusion.com?utm_source=github&utm_medium=referral&utm_campaign=SDNext)

> [!TIP]
> - Server can run without virtual environment,  
  Recommended to use `VENV` to avoid library version conflicts with other applications  
> - **nVidia/CUDA** / **AMD/ROCm** / **Intel/OneAPI** are auto-detected if present and available,  
  For any other use case such as **DirectML**, **ONNX/Olive**, **OpenVINO** specify required parameter explicitly  
  or wrong packages may be installed as installer will assume CPU-only environment  
> - Full startup sequence is logged in `sdnext.log`,  
  so if you encounter any issues, please check it first  

### Run

Once SD.Next is installed, simply run `webui.ps1` or `webui.bat` (*Windows*) or `webui.sh` (*Linux or MacOS*)

Below is partial list of all available parameters, run `webui --help` for the full list:

    Server options:
      --config CONFIG                  Use specific server configuration file, default: config.json
      --ui-config UI_CONFIG            Use specific UI configuration file, default: ui-config.json
      --medvram                        Split model stages and keep only active part in VRAM, default: False
      --lowvram                        Split model components and keep only active part in VRAM, default: False
      --ckpt CKPT                      Path to model checkpoint to load immediately, default: None
      --vae VAE                        Path to VAE checkpoint to load immediately, default: None
      --data-dir DATA_DIR              Base path where all user data is stored, default:
      --models-dir MODELS_DIR          Base path where all models are stored, default: models
      --share                          Enable UI accessible through Gradio site, default: False
      --insecure                       Enable extensions tab regardless of other options, default: False
      --listen                         Launch web server using public IP address, default: False
      --auth AUTH                      Set access authentication like "user:pwd,user:pwd""
      --autolaunch                     Open the UI URL in the system's default browser upon launch
      --docs                           Mount Gradio docs at /docs, default: False
      --no-hashing                     Disable hashing of checkpoints, default: False
      --no-metadata                    Disable reading of metadata from models, default: False
      --no-download                    Disable download of default model, default: False
      --backend {original,diffusers}   force model pipeline type

    Setup options:
      --debug                          Run installer with debug logging, default: False
      --reset                          Reset main repository to latest version, default: False
      --upgrade                        Upgrade main repository to latest version, default: False
      --requirements                   Force re-check of requirements, default: False
      --quick                          Run with startup sequence only, default: False
      --use-directml                   Use DirectML if no compatible GPU is detected, default: False
      --use-openvino                   Use Intel OpenVINO backend, default: False
      --use-ipex                       Force use Intel OneAPI XPU backend, default: False
      --use-cuda                       Force use nVidia CUDA backend, default: False
      --use-rocm                       Force use AMD ROCm backend, default: False
      --use-xformers                   Force use xFormers cross-optimization, default: False
      --skip-requirements              Skips checking and installing requirements, default: False
      --skip-extensions                Skips running individual extension installers, default: False
      --skip-git                       Skips running all GIT operations, default: False
      --skip-torch                     Skips running Torch checks, default: False
      --skip-all                       Skips running all checks, default: False
      --experimental                   Allow unsupported versions of libraries, default: False
      --reinstall                      Force reinstallation of all requirements, default: False
      --safe                           Run in safe mode with no user extensions


## Notes

### **Extensions**

SD.Next comes with several extensions pre-installed:

- [ControlNet](https://github.com/Mikubill/sd-webui-controlnet)
- [Agent Scheduler](https://github.com/ArtVentureX/sd-webui-agent-scheduler)
- [Image Browser](https://github.com/AlUlkesh/stable-diffusion-webui-images-browser)

### **Collab**

- We'd love to have additional maintainers with full admin rights. If you're interested, ping us!  
- In addition to general cross-platform code, desire is to have a lead for each of the main platforms.
This should be fully cross-platform, but we'd really love to have additional contributors and/or maintainers to join and help lead the efforts on different platforms.

### **Goals**

This project started as a fork from [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) and it grew significantly since then,  
but although it diverged considerably, any substantial features to original work is ported to this repository as well.

The idea behind the fork is to enable latest technologies and advances in text-to-image generation.

*Sometimes this is not the same as "as simple as possible to use".*

General goals:

- Multi-model
  - Enable usage of as many as possible txt2img and img2img generative models  
- Cross-platform
  - Create uniform experience while automatically managing any platform specific differences
- Performance
  - Enable best possible performance on all platforms
- Ease-of-Use
  - Automatically handle all requirements, dependencies, flags regardless of platform
  - Integrate all best options for uniform out-of-the-box experience without the need to tweak anything manually
- Look-and-Feel
  - Create modern, intuitive and clean UI
- Up-to-Date
  - Keep code up to date with latest advanced in text-to-image generation
=======
## Documentation

The documentation was moved from this README over to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

For the purposes of getting Google and other search engines to crawl the wiki, here's a link to the (not for humans) [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits

<<<<<<< HEAD
- Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Additional credits are listed in [Credits](https://github.com/AUTOMATIC1111/stable-diffusion-webui/#credits)
- Licenses for modules are listed in [Licenses](html/licenses.html)

### **Docs**

If you're unsure how to use a feature, best place to start is [Wiki](https://github.com/vladmandic/automatic/wiki) and if its not there,  
check [ChangeLog](CHANGELOG.md) for when feature was first introduced as it will always have a short note on how to use it  

- [Wiki](https://github.com/vladmandic/automatic/wiki)
- [ReadMe](README.md)
- [ToDo](TODO.md)  
- [ChangeLog](CHANGELOG.md)
- [CLI Tools](cli/README.md)

### **Sponsors**

<div align="center">
<!-- sponsors --><a href="https://github.com/allangrant"><img src="https://github.com/allangrant.png" width="60px" alt="Allan Grant" /></a><a href="https://github.com/BrentOzar"><img src="https://github.com/BrentOzar.png" width="60px" alt="Brent Ozar" /></a><a href="https://github.com/inktomi"><img src="https://github.com/inktomi.png" width="60px" alt="Matthew Runo" /></a><a href="https://github.com/HELLO-WORLD-SAS"><img src="https://github.com/HELLO-WORLD-SAS.png" width="60px" alt="HELLO WORLD SAS" /></a><a href="https://github.com/4joeknight4"><img src="https://github.com/4joeknight4.png" width="60px" alt="" /></a><a href="https://github.com/SaladTechnologies"><img src="https://github.com/SaladTechnologies.png" width="60px" alt="Salad Technologies" /></a><a href="https://github.com/mantzaris"><img src="https://github.com/mantzaris.png" width="60px" alt="a.v.mantzaris" /></a><!-- sponsors -->
</div>

<br>
=======
- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instruct pix2pix - Tim Brooks (star), Aleksander Holynski (star), Alexei A. Efros (no star) - https://github.com/timothybrooks/instruct-pix2pix
- Security advice - RyotaK
- UniPC sampler - Wenliang Zhao - https://github.com/wl-zhao/UniPC
- TAESD - Ollin Boer Bohan - https://github.com/madebyollin/taesd
- LyCORIS - KohakuBlueleaf
- Restart sampling - lambertae - https://github.com/Newbeeer/diffusion_restart_sampling
- Hypertile - tfernd - https://github.com/tfernd/HyperTile
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
