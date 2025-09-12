import modal
import os
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
    results_dir = os.path.join(repo, "results")

    subprocess.run(
        [
            "python",
            "scripts/inference_partcrafter.py",
            "--image_path",
            "assets/images/np3_2f6ab901c5a84ed6bbdf85a67b22a2ee.png",
            "--num_parts",
            "3",
            "--tag",
            "robot",
            "--render",
        ],
        cwd=repo,
        check=True,
    )

    export_dir = "/partcrafter/results"
    os.makedirs(export_dir, exist_ok=True)

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    shutil.copytree(results_dir, export_dir)

    print("Copied results to volume:", export_dir)


@app.local_entrypoint()
def main():
    run_inference.remote()
