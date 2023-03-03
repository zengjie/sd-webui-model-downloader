import os

import gradio as gr
import requests

from modules import scripts, script_callbacks, ui


def download_model(model_url, model_type, progress=gr.Progress()):
    # Download the model and update progress
    response = requests.get(model_url, stream=True)
    total_length = response.headers.get('content-length')
    block_size = 1024 * 1024 # 1 MB

    progress.tqdm(unit='iB', unit_scale=True, total=int(total_length) / block_size)
    
    model_filename = model_url.split("/")[-1]
    model_path = os.path.join(scripts.basedir(), "models", model_filename)
    with open(model_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress.update(len(data))
            f.write(data)

def add_tab():
    with gr.Blocks("Model Downloader") as tab:
        with gr.Row():
            with gr.Column():
                # Write a simple interface to download models
                model_url = gr.Text("URL")
                model_type = gr.Dropdown(["checkpoint", "controlnet"], label="Type", default="checkpoint")
                download_button = gr.Button("Download")

        # Start a async task to download the model
        download_button.click(download_model, inputs=[model_url, model_type])

    return [(tab, "Model Downloader", "model_downloader")]


script_callbacks.on_ui_tabs(add_tab)