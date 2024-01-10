#!/usr/bin/env python

from __future__ import annotations

import os
import random
import gc
import toml
import gradio as gr
import numpy as np
import utils
import torch
import json
import PIL.Image
import base64
import safetensors
from io import BytesIO
from typing import Tuple
from datetime import datetime
from PIL import PngImagePlugin
import gradio_user_history as gr_user_history
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from lora_diffusers import LoRANetwork, create_network_from_weights
from diffusers.models import AutoencoderKL
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
)

DESCRIPTION = "Animagine XL 3.0"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU. </p>"
IS_COLAB = utils.is_google_colab() or os.getenv("IS_COLAB") == "1"
MAX_SEED = np.iinfo(np.int32).max
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "512"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"

MODEL = os.getenv("MODEL", "https://huggingface.co/Linaqruf/animagine-xl-3.0/blob/main/animagine-xl-3.0.safetensors")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionXLPipeline.from_single_file if MODEL.endswith(".safetensors") else StableDiffusionXLPipeline.from_pretrained
    
    pipe = pipeline(
        MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        use_auth_token=HF_TOKEN,
        variant="fp16",
    )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
else:
    pipe = None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_image_path(base_path: str):
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    for ext in extensions:
        image_path = base_path + ext
        if os.path.exists(image_path):
            return image_path
    return None


def update_selection(selected_state: gr.SelectData):
    lora_repo = sdxl_loras[selected_state.index]["repo"]
    lora_weight = sdxl_loras[selected_state.index]["multiplier"]
    updated_selected_info = f"{lora_repo}"

    return (
        updated_selected_info,
        selected_state,
        lora_weight,
    )


def parse_aspect_ratio(aspect_ratio):
    if aspect_ratio == "Custom":
        return None, None
    width, height = aspect_ratio.split(" x ")
    return int(width), int(height)


def aspect_ratio_handler(aspect_ratio, custom_width, custom_height):
    if aspect_ratio == "Custom":
        return custom_width, custom_height
    else:
        width, height = parse_aspect_ratio(aspect_ratio)
        return width, height


def create_network(text_encoders, unet, state_dict, multiplier, device):
    network = create_network_from_weights(
        text_encoders,
        unet,
        state_dict,
        multiplier,
    )
    network.load_state_dict(state_dict)
    network.to(device, dtype=unet.dtype)
    network.apply_to(multiplier=multiplier)

    return network


def get_scheduler(scheduler_config, name):
    scheduler_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(
            scheduler_config
        ),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_map.get(name, lambda: None)()


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def preprocess_prompt(
    style_dict,
    style_name: str,
    positive: str,
    negative: str = "",
    add_style: bool = True,
) -> Tuple[str, str]:
    p, n = style_dict.get(style_name, style_dict["(None)"])

    if add_style and positive.strip():
        formatted_positive = p.format(prompt=positive)
    else:
        formatted_positive = positive

    combined_negative = n + negative
    return formatted_positive, combined_negative


def common_upscale(samples, width, height, upscale_method):
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode=upscale_method
    )


def upscale(samples, upscale_method, scale_by):
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = common_upscale(samples, width, height, upscale_method)
    return s


def load_and_convert_thumbnail(model_path: str):
    with safetensors.safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    if "modelspec.thumbnail" in metadata:
        base64_data = metadata["modelspec.thumbnail"]
        prefix, encoded = base64_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = PIL.Image.open(BytesIO(image_data))
        return image
    return None

def load_wildcard_files(wildcard_dir):
    wildcard_files = {}
    for file in os.listdir(wildcard_dir):
        if file.endswith(".txt"):
            key = f"__{file.split('.')[0]}__"  # Create a key like __character__
            wildcard_files[key] = os.path.join(wildcard_dir, file)
    return wildcard_files

def get_random_line_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            return ""
        return random.choice(lines).strip()

def add_wildcard(prompt, wildcard_files):
    for key, file_path in wildcard_files.items():
        if key in prompt:
            wildcard_line = get_random_line_from_file(file_path)
            prompt = prompt.replace(key, wildcard_line)
    return prompt

def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    custom_width: int = 1024,
    custom_height: int = 1024,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 28,
    use_lora: bool = False,
    lora_weight: float = 1.0,
    selected_state: str = "",
    sampler: str = "Euler a",
    aspect_ratio_selector: str = "896 x 1152",
    style_selector: str = "(None)",
    quality_selector: str = "Standard",
    use_upscaler: bool = False,
    upscaler_strength: float = 0.5,
    upscale_by: float = 1.5,
    add_quality_tags: bool = True,
    profile: gr.OAuthProfile | None = None,
    progress=gr.Progress(track_tqdm=True),
) -> PIL.Image.Image:
    generator = seed_everything(seed)

    network = None
    network_state = {"current_lora": None, "multiplier": None}

    width, height = aspect_ratio_handler(
        aspect_ratio_selector,
        custom_width,
        custom_height,
    )

    prompt = add_wildcard(prompt, wildcard_files)

    
    prompt, negative_prompt = preprocess_prompt(
        quality_prompt, quality_selector, prompt, negative_prompt, add_quality_tags
    )
    prompt, negative_prompt = preprocess_prompt(
        styles, style_selector, prompt, negative_prompt
    )

    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)
        
    if use_lora:
        if not selected_state:
            raise Exception("You must Select a LoRA")
        repo_name = sdxl_loras[selected_state.index]["repo"]
        full_path_lora = saved_names[selected_state.index]
        weight_name = sdxl_loras[selected_state.index]["weights"]

        lora_sd = load_file(full_path_lora)
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

        if network_state["current_lora"] != repo_name:
            network = create_network(
                text_encoders,
                pipe.unet,
                lora_sd,
                lora_weight,
                device,
            )
            network_state["current_lora"] = repo_name
            network_state["multiplier"] = lora_weight
        elif network_state["multiplier"] != lora_weight:
            network = create_network(
                text_encoders,
                pipe.unet,
                lora_sd,
                lora_weight,
                device,
            )
            network_state["multiplier"] = lora_weight
    else:
        if network:
            network.unapply_to()
            network = None
            network_state = {
                "current_lora": None,
                "multiplier": None,
            }

    backup_scheduler = pipe.scheduler
    pipe.scheduler = get_scheduler(pipe.scheduler.config, sampler)

    if use_upscaler:
        upscaler_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)

    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": f"{width} x {height}",
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "sampler": sampler,
        "sdxl_style": style_selector,
        "add_quality_tags": add_quality_tags,
        "quality_tags": quality_selector,
    }

    if use_lora:
        metadata["use_lora"] = {"selected_lora": repo_name, "multiplier": lora_weight}
    else:
        metadata["use_lora"] = None

    if use_upscaler:
        new_width = int(width * upscale_by)
        new_height = int(height * upscale_by)
        metadata["use_upscaler"] = {
            "upscale_method": "nearest-exact",
            "upscaler_strength": upscaler_strength,
            "upscale_by": upscale_by,
            "new_resolution": f"{new_width} x {new_height}",
        }
    else:
        metadata["use_upscaler"] = None

    print(json.dumps(metadata, indent=4))

    try:
        if use_upscaler:
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
            upscaled_latents = upscale(latents, "nearest-exact", upscale_by)
            image = upscaler_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=upscaler_strength,
                generator=generator,
                output_type="pil",
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images[0]
        if network:
            network.unapply_to()
            network = None
        if profile is not None:
            gr_user_history.save_image(
                label=prompt,
                image=image,
                profile=profile,
                metadata=metadata,
            )
        if image and IS_COLAB:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_directory = "./outputs"
            os.makedirs(output_directory, exist_ok=True)
            filename = f"image_{current_time}.png"
            filepath = os.path.join(output_directory, filename)

            # Convert metadata to a string and save as a text chunk in the PNG
            metadata_str = json.dumps(metadata)
            info = PngImagePlugin.PngInfo()
            info.add_text("metadata", metadata_str)
            image.save(filepath, "PNG", pnginfo=info)
            print(f"Image saved as {filepath} with metadata")

        return image, metadata

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        if network:
            network.unapply_to()
            network = None
        if use_lora:
            del lora_sd, text_encoders
        if use_upscaler:
            del upscaler_pipe
        pipe.scheduler = backup_scheduler
        free_memory()


examples = [
    "1girl, arima kana, oshi no ko, solo, idol, idol clothes, one eye closed, red shirt, black skirt, black headwear, gloves, stage light, singing, open mouth, crowd, smile, pointing at viewer",
    "1girl, c.c., code geass, white shirt, long sleeves, turtleneck, sitting, looking at viewer, eating, pizza, plate, fork, knife, table, chair, table, restaurant, cinematic angle, cinematic lighting",
    "1girl, sakurauchi riko, \(love live\), queen hat, noble coat, red coat, noble shirt, sitting, crossed legs, gentle smile, parted lips, throne, cinematic angle",
    "1girl, amiya \(arknights\), arknights, dirty face, outstretched hand, close-up, cinematic angle, foreshortening, dark, dark background",
    "A boy and a girl, Emiya Shirou and Artoria Pendragon from fate series, having their breakfast in the dining room. Emiya Shirou wears white t-shirt and jacket. Artoria Pendragon wears white dress with blue neck ribbon. Rice, soup, and minced meats are served on the table. They look at each other while smiling happily",
]

quality_prompt_list = [
    {
        "name": "(None)",
        "prompt": "{prompt}",
        "negative_prompt": "nsfw, lowres, ",
    },
    {
        "name": "Standard",
        "prompt": "{prompt}, masterpiece, best quality",
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, ",
    },
    {
        "name": "Light",
        "prompt": "{prompt}, (masterpiece), best quality, perfect face",
        "negative_prompt": "nsfw, (low quality, worst quality:1.2), 3d, watermark, signature, ugly, poorly drawn, ",
    },
    {
        "name": "Heavy",
        "prompt": "{prompt}, (masterpiece), (best quality), (ultra-detailed), illustration, disheveled hair, perfect composition, moist skin, intricate details, earrings",
        "negative_prompt": "nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality, ",
    },
]

sampler_list = [
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DPM++ 2M SDE Karras",
    "Euler",
    "Euler a",
    "DDIM",
]

aspect_ratios = [
    "1024 x 1024",
    "1152 x 896",
    "896 x 1152",
    "1216 x 832",
    "832 x 1216",
    "1344 x 768",
    "768 x 1344",
    "1536 x 640",
    "640 x 1536",
    "Custom",
]

style_list = [
    {
        "name": "(None)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "{prompt}, cinematic still, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "nsfw, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "{prompt}, cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "nsfw, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "{prompt}, anime artwork, anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "nsfw, photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "{prompt}, manga style, vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "nsfw, ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "{prompt}, concept art, digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "nsfw, photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "{prompt}, pixel-art, low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "nsfw, sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "{prompt}, ethereal fantasy concept art, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "nsfw, photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "{prompt}, neonpunk style, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "nsfw, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "{prompt}, professional 3d model, octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "nsfw, ugly, deformed, noisy, low poly, blurry, painting",
    },
]

thumbnail_cache = {}

with open("lora.toml", "r") as file:
    data = toml.load(file)

sdxl_loras = []
saved_names = []
for item in data["data"]:
    model_path = hf_hub_download(item["repo"], item["weights"], token=HF_TOKEN)
    saved_names.append(model_path)  # Store the path in saved_names

    if model_path not in thumbnail_cache:
        thumbnail_image = load_and_convert_thumbnail(model_path)
        thumbnail_cache[model_path] = thumbnail_image
    else:
        thumbnail_image = thumbnail_cache[model_path]

    sdxl_loras.append(
        {
            "image": thumbnail_image,  # Storing the PIL image object
            "title": item["title"],
            "repo": item["repo"],
            "weights": item["weights"],
            "multiplier": item.get("multiplier", "1.0"),
        }
    )

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
quality_prompt = {
    k["name"]: (k["prompt"], k["negative_prompt"]) for k in quality_prompt_list
}

# saved_names = [
#     hf_hub_download(item["repo"], item["weights"], token=HF_TOKEN)
#     for item in sdxl_loras
# ]

wildcard_files = load_wildcard_files("wildcard")

with gr.Blocks(css="style.css", theme="NoCrypt/miku@1.2.1") as demo:
    title = gr.HTML(
        f"""<h1><span>{DESCRIPTION}</span></h1>""",
        elem_id="title",
    )
    gr.Markdown(
        f"""Gradio demo for [cagliostrolab/animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0)""",
        elem_id="subtitle",
    )
    gr.Markdown(
        f"""Prompting is a bit different in this iteration, we train the model like this:
        ```
        1girl/1boy, character name, from what series, everything else in any order. 
        ```
        Prompting Tips
        ```
        1. Quality Tags: `masterpiece, best quality, high quality, normal quality, worst quality, low quality`
        2. Year Tags: `oldest, early, mid, late, newest`
        3. Rating tags: `rating: general, rating: sensitive, rating: questionable, rating: explicit, nsfw`
        4. Escape character: `character name \(series\)`
        5. Recommended settings: `Euler a, cfg 5-7, 25-28 steps`
        6. It's recommended to use the exact danbooru tags for more accurate result
        7. To use character wildcard, add this syntax to the prompt `__character__`.
        ```
        """,
        elem_id="subtitle",
    )   
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    selected_state = gr.State()
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Txt2img"):
                with gr.Group():
                    prompt = gr.Text(
                        label="Prompt",
                        max_lines=5,
                        placeholder="Enter your prompt",
                    )
                    negative_prompt = gr.Text(
                        label="Negative Prompt",
                        max_lines=5,
                        placeholder="Enter a negative prompt",
                    )
                    with gr.Accordion(label="Quality Tags", open=True):
                        add_quality_tags = gr.Checkbox(label="Add Quality Tags", value=True)
                        quality_selector = gr.Dropdown(
                            label="Quality Tags Presets",
                            interactive=True,
                            choices=list(quality_prompt.keys()),
                            value="Standard",
                        )
                    with gr.Row():
                        use_lora = gr.Checkbox(label="Use LoRA", value=False)
                with gr.Group(visible=False) as lora_group:
                    selector_info = gr.Text(
                        label="Selected LoRA",
                        max_lines=1,
                        value="No LoRA selected.",
                    )
                    lora_selection = gr.Gallery(
                        value=[(item["image"], item["title"]) for item in sdxl_loras],
                        label="Animagine XL 2.0 LoRA",
                        show_label=False,
                        columns=2,
                        show_share_button=False,
                    )
                    lora_weight = gr.Slider(
                        label="Multiplier",
                        minimum=-2,
                        maximum=2,
                        step=0.05,
                        value=1,
                    )
            with gr.Tab("Advanced Settings"):
                with gr.Group():
                    style_selector = gr.Radio(
                        label="Style Preset",
                        container=True,
                        interactive=True,
                        choices=list(styles.keys()),
                        value="(None)",
                    )
                with gr.Group():
                    aspect_ratio_selector = gr.Radio(
                        label="Aspect Ratio",
                        choices=aspect_ratios,
                        value="896 x 1152",
                        container=True,
                    )
                with gr.Group():
                    use_upscaler = gr.Checkbox(label="Use Upscaler", value=False)
                    with gr.Row() as upscaler_row:
                        upscaler_strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.55,
                            visible=False,
                        )
                        upscale_by = gr.Slider(
                            label="Upscale by",
                            minimum=1,
                            maximum=1.5,
                            step=0.1,
                            value=1.5,
                            visible=False,
                        )
                with gr.Group(visible=False) as custom_resolution:
                    with gr.Row():
                        custom_width = gr.Slider(
                            label="Width",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        custom_height = gr.Slider(
                            label="Height",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                with gr.Group():
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=sampler_list,
                        interactive=True,
                        value="Euler a",
                    )
                with gr.Group():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                    )

                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Group():
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=1,
                            maximum=12,
                            step=0.1,
                            value=7.0,
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28,
                        )

            with gr.Tab("Past Generation"):
                gr_user_history.render()
        with gr.Column(scale=3):
            with gr.Blocks():
                run_button = gr.Button("Generate", variant="primary")
            result = gr.Image(label="Result", show_label=False)
            with gr.Accordion(label="Generation Parameters", open=False):
                gr_metadata = gr.JSON(label="Metadata", show_label=False)
            gr.Examples(
                examples=examples,
                inputs=prompt,
                outputs=[result, gr_metadata],
                fn=generate,
                cache_examples=CACHE_EXAMPLES,
            )

    lora_selection.select(
        update_selection,
        outputs=[
            selector_info,
            selected_state,
            lora_weight,
        ],
        queue=False,
        show_progress=False,
    )
    use_lora.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_lora,
        outputs=lora_group,
        queue=False,
        api_name=False,
    )
    use_upscaler.change(
        fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=use_upscaler,
        outputs=[upscaler_strength, upscale_by],
        queue=False,
        api_name=False,
    )
    aspect_ratio_selector.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio_selector,
        outputs=custom_resolution,
        queue=False,
        api_name=False,
    )

    inputs = [
        prompt,
        negative_prompt,
        seed,
        custom_width,
        custom_height,
        guidance_scale,
        num_inference_steps,
        use_lora,
        lora_weight,
        selected_state,
        sampler,
        aspect_ratio_selector,
        style_selector,
        quality_selector,
        use_upscaler,
        upscaler_strength,
        upscale_by,
        add_quality_tags
    ]

    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=[result, gr_metadata],
        api_name=False,
    )
demo.queue(max_size=20).launch(debug=IS_COLAB, share=IS_COLAB)
