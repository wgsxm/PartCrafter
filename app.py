import argparse
import os
import sys
from glob import glob
import time
from typing import Any, Union
import gradio as gr
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed
import random
import string
import secrets

from src.utils.data_utils import get_colored_mesh_composition, scene_to_parts, load_surfaces
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

MAX_NUM_PARTS = 16
device = "cuda"
dtype = torch.float16

# download pretrained weights
partcrafter_weights_dir = "pretrained_weights/PartCrafter"
rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir)
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

# init rmbg model for background removal
rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
rmbg_net.eval() 

# init tripoSG pipeline
pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: str,
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> trimesh.Scene:

    if rmbg:
        img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        img_pil = Image.open(image_input)
    start_time = time.time()
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    for i in range(len(outputs)):
        if outputs[i] is None:
            # If the generated mesh is None (decoding error), use a dummy mesh
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
    return outputs, img_pil

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(secrets.choice(letters) for i in range(length))

def generate_mesh(image, num_parts, seed, num_tokens, num_inference_steps, guidance_scale, rmbg):
    # Generate random seed if seed is 0
    if seed == 0:
        seed = secrets.randbelow(9999) + 1  # Generates from 1 to 9999
    
    set_seed(seed)
    
    temp_dir = "results"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    folder_name = generate_random_string(10)
    export_dir = os.path.join(temp_dir, folder_name)
    os.makedirs(export_dir, exist_ok=True)
    
    image_path = os.path.join(export_dir, "input.png")
    image.save(image_path)

    # run inference
    outputs, processed_image = run_triposg(
        pipe,
        image_input=image_path,
        num_parts=num_parts,
        rmbg_net=rmbg_net,
        seed=seed,  # Now seed will never be 0
        num_tokens=num_tokens,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        rmbg=rmbg,
        dtype=dtype,
        device=device,
    )
    
    merged_mesh = get_colored_mesh_composition(outputs)
    mesh_path = os.path.join(export_dir, "object.glb")
    merged_mesh.export(mesh_path)
    print(f"Generated {len(outputs)} parts and saved to {export_dir}")
    
    return mesh_path

with gr.Blocks() as demo:
    gr.Markdown("# PartCrafter")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            num_parts = gr.Slider(minimum=1, maximum=MAX_NUM_PARTS, value=4, step=1, label="Number of Parts")
            seed = gr.Slider(minimum=0, maximum=10000, value=0, step=1, label="Seed")
            num_tokens = gr.Slider(minimum=256, maximum=2048, value=1024, step=256, label="Number of Tokens")
            num_inference_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Number of Inference Steps")
            guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="Guidance Scale")
            rmbg = gr.Checkbox(label="Remove Background", value=False)
            generate_button = gr.Button("Generate")
        with gr.Column():
            output_model = gr.Model3D(label="Generated Model", display_mode="solid")

    generate_button.click(
        fn=generate_mesh,
        inputs=[image_input, num_parts, seed, num_tokens, num_inference_steps, guidance_scale, rmbg],
        outputs=output_model,
    )

if __name__ == "__main__":
    demo.launch()
