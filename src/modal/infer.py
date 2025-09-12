import glob
import modal
import os
import re
import shutil
import subprocess


app = modal.App("playcrafter")

base_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .add_local_dir(".", "/root/PartCrafter", copy=True)
    .apt_install(
        "libegl1",
        "libegl1-mesa",
        "libgl1-mesa-dev",
        "libglu1-mesa",
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",
    )
    .run_commands("cd /root/PartCrafter && bash settings/setup.sh")
)

partcrafter_volume = modal.Volume.from_name(
    "partcrafter",
    create_if_missing=True,
)


@app.function(
    image=base_image,
    volumes={"/partcrafter": partcrafter_volume},
    gpu="H100",
    timeout=600,
    memory=12_288,
    ephemeral_disk=524_288,
)
def run_inference():
    repo = "/root/PartCrafter"
    assets_dir = os.path.join(repo, "assets", "images")
    results_dir = os.path.join(repo, "results")

    imgs = sorted(
        glob.glob(os.path.join(assets_dir, "*.png"))
        + glob.glob(os.path.join(assets_dir, "*.jpg"))
        + glob.glob(os.path.join(assets_dir, "*.jpeg"))
    )

    if not imgs:
        print(f"No images found in {assets_dir}")
        return
    else:
        print(f"Found {len(imgs)} images")

    for img in imgs:
        base = os.path.splitext(os.path.basename(img))[0]
        m = re.match(r"^np(\d+)_", base)
        num_parts = m.group(1) if m else "6"
        tag = base

        cmd = [
            "python",
            "scripts/inference_partcrafter.py",
            "--image_path", os.path.relpath(img, repo),
            "--num_parts", str(num_parts),
            "--tag", tag,
            "--render",
            "--rmbg",
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=repo, check=True)

    export_dir = "/partcrafter/results"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    shutil.copytree(results_dir, export_dir)

    print("Copied results to volume:", export_dir)


@app.local_entrypoint()
def main():
    run_inference.remote()
