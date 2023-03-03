import os

import gradio as gr
import requests
import tqdm

from modules import scripts, script_callbacks, ui
from modules.call_queue import wrap_gradio_call


def download_model(model_url, model_type, progress=gr.Progress(track_tqdm=True)):
    # Download the model and update progress
    response = requests.get(model_url, stream=True)
    total_length = int(response.headers.get('content-length'))
    block_size = 1024 * 1024 # 1 MB

    p = tqdm.tqdm(unit='iB', unit_scale=True, total=total_length, desc='Downloading model')
    
    model_filename = model_url.split("/")[-1]
    model_path = os.path.join(scripts.basedir(), "models", model_filename)
    with open(model_path, "wb") as f:
        for data in response.iter_content(block_size):
            p.update(len(data))
            progress(0.5)
            f.write(data)
    p.close()

    return f"Downloaded model to: {model_path}"

def add_tab():
    with gr.Blocks("Model Downloader") as tab:
        with gr.Row():
            with gr.Column():
                # Write a simple interface to download models
                model_url = gr.Text("URL")
                model_type = gr.Dropdown(["checkpoint", "controlnet"], label="Type", default="checkpoint")
                download_button = gr.Button("Download")
                output = gr.Textbox("Output", lines=5)


        # Start a async task to download the model
        download_button.click(
            wrap_gradio_call(download_model), 
            inputs=[model_url, model_type], 
            outputs=output, 
            show_progress=True,
        )

    return [(tab, "Model Downloader", "model_downloader")]


script_callbacks.on_ui_tabs(add_tab)