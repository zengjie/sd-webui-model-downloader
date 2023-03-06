import os

import gradio as gr
import requests
import tqdm

from modules import scripts, script_callbacks, progress
from modules.call_queue import wrap_gradio_call, wrap_gradio_gpu_call


PREDEFINED_MODELS = [
    {
        "name": "v2-1_768-ema-pruned.safetensors",
        "description": "Stable Diffusion v2.1",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors",
        "type": "Stable-diffusion",
    },
    {
        "name": "civitai-DreamShaper-v3.32-pruned.safetensors",
        "description": "DreamShaper v3.32 Pruned SafeTensor",
        "url": "https://civitai.com/api/download/models/5636?type=Pruned%20Model&format=SafeTensor",
        "type": "Stable-diffusion",
    },
    {
        "name": "control_sd15_normal.pth",
        "description": "ConttrolNet SD v1.5 Normal",
        "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth",
        "type": "ControlNet",
    },
    {
        "name": "control_sd15_scribble.pth",
        "description": "ConttrolNet SD v1.5 Scribble",
        "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth",
        "type": "ControlNet",
    },
    {
        "name": "control_sd15_seg.pth",
        "description": "ConttrolNet SD v1.5 Segmentation",
        "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth",
        "type": "ControlNet",
    },
]


def download_model(model_url, model_type, model_filename, progress=gr.Progress(track_tqdm=True)):
    # Download the model and save it to the models folder, follow redirects
    response = requests.get(model_url, stream=True, allow_redirects=True)
    total_length = int(response.headers.get("content-length"))
    block_size = 1024 * 1024  # 1 MB

    p = tqdm.tqdm(
        unit="iB", unit_scale=True, total=total_length, desc="Downloading"
    )

    if not model_filename:
        model_filename = model_url.split("/")[-1]
    model_path = os.path.join(scripts.basedir(), "models", model_type, model_filename)
    with open(model_path, "wb") as f:
        for data in response.iter_content(block_size):
            p.update(len(data))
            f.write(data)
    p.close()

    return f"Downloaded"


def add_tab():
    with gr.Blocks("Model Downloader") as tab:
        gr.HTML(
            "Download models from the internet and save them to the models folder."
        )

        with gr.Box():
            # List predefined models and provide a download button
            for model in PREDEFINED_MODELS:
                with gr.Row():
                    with gr.Column(scale=30):
                        gr.HTML(f"<pre>{model['name']}</pre>")
                    with gr.Column(scale=40):
                        gr.Markdown(model["description"])
                    with gr.Column(scale=10):
                        download_button = gr.Button("Download")
                    with gr.Column(scale=20):
                        download_result = gr.Textbox("", label=">", lines=2, interactive=False)

                model_url = gr.State(model["url"])
                model_type = gr.State(model["type"])
                model_filename = gr.State(model["name"])
                download_button.click(
                    fn=download_model,
                    inputs=[model_url, model_type, model_filename],
                    outputs=download_result,
                    queue=True,
                )


        with gr.Accordion("Manual Download", open=False):
                # Write a simple interface to download models
                model_url = gr.Text(
                    label="URL", value="https://speed.hetzner.de/100MB.bin"
                )
                model_type = gr.Dropdown(
                    ["Stable-diffusion", "ControlNet"], label="Type", value="Stable-diffusion"
                )
                model_filename = gr.Text(label="Filename", value="")
                download_button = gr.Button("Download")
                download_result = gr.Textbox("", label="Output")

                download_button.click(
                    fn=download_model,
                    inputs=[model_url, model_type, model_filename],
                    outputs=download_result,
                    queue=True,
                )

    return [(tab, "Model Downloader", "model_downloader")]


script_callbacks.on_ui_tabs(add_tab)
