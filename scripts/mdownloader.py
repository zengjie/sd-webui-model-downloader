import os
import json
import threading
import enum
import time
import shutil

import gradio as gr
import requests

is_testing = False
try:
    from modules import script_callbacks
except ImportError:
    is_testing = True


MAX_ROWS = 50

MODEL_TYPES = [
    "Stable-diffusion",
    "ControlNet",
    "ESRGAN",
    "GFPGAN",
    "LDSR",
    "Lora",
    "hypernetworks"
    "VAE",
    "AVE-approx",
    "deepbooru",
]

DOWNLOAD_TASKS = {}

PREDEFINED_MODELS = []

INITIAL_INDEX_URL = "https://raw.githubusercontent.com/zengjie/sd-webui-model-downloader/main/data/index.json"

if is_testing:
    PREDEFINED_MODELS += [
        {
            "name": "demo.bin",
            "description": "Download a demo model",
            "url": "http://speedtest.ftp.otenet.gr/files/test10Mb.db",
            "type": "demo",
        },
    ]

    INITIAL_INDEX_URL = "./data/index.json"


def convert_bytes(num):
    """
    This function takes a number of bytes as input and returns a string
    representing the equivalent number of KB, MB, GB, or TB.
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def get_model_remote_size(model_url):
    result = None

    # load the size from mdownloader_cache.json file if exists
    if os.path.exists("mdownloader_cache.json"):
        with open("mdownloader_cache.json", "r") as f:
            cache = json.load(f)
            if model_url in cache:
                return cache[model_url]

    try:
        # Get the size of the model file
        response = requests.head(model_url, allow_redirects=True)

        # if HEAD is not supported, use GET
        if response.status_code != 200:
            response = requests.get(model_url, stream=True, allow_redirects=True)
            # Close the connection to avoid memory leak
            response.close()

        if "Content-Length" in response.headers:
            result = int(response.headers["Content-Length"])
        else:
            print(model_url, "does not have Content-Length header.")
            result = None
    except Exception as e:
        print("Failed to get size of", model_url, e)
        result = None

    # memorize the size to mdownloader_cache.json file to avoid repeated requests
    if result:
        cache = {}
        if os.path.exists("mdownloader_cache.json"):
            with open("mdownloader_cache.json", "r") as f:
                cache = json.load(f)
        cache[model_url] = result
        with open("mdownloader_cache.json", "w") as f:
            json.dump(cache, f)

    return result


def get_model_local_size(model_type, model_filename):
    base_dir = "."
    model_path = os.path.join(base_dir, "models", model_type, model_filename)
    if os.path.exists(model_path):
        return os.path.getsize(model_path)
    else:
        return None


def add_download_task(model_type, model_filename):
    DOWNLOAD_TASKS[model_type + "/" + model_filename] = True


def stop_download_task(model_type, model_filename):
    DOWNLOAD_TASKS[model_type + "/" + model_filename] = False
    while 1:
        if not is_download_task_exists(model_type, model_filename):
            break
        time.sleep(0.1)


def should_stop_download_task(model_type, model_filename):
    return not DOWNLOAD_TASKS[model_type + "/" + model_filename]


def remove_download_task(model_type, model_filename):
    DOWNLOAD_TASKS.pop(model_type + "/" + model_filename, None)


def is_download_task_exists(model_type, model_filename):
    return (model_type + "/" + model_filename) in DOWNLOAD_TASKS


def _download_model_worker(model_url, model_type, model_filename):
    add_download_task(model_type, model_filename)

    # Download the model and save it to the models folder, follow redirects
    response = requests.get(model_url, stream=True, allow_redirects=True)
    total_length = int(response.headers.get("content-length"))
    block_size = 1024 * 1024  # 1 MB

    # Get the filename from the URL if not provided
    if not model_filename:
        model_filename = model_url.split("/")[-1]

    # Get the base directory
    base_dir = "."
    model_path = os.path.join(base_dir, "models", model_type, model_filename)

    # Create the directory if it doesn't exist
    parent_dir = os.path.dirname(model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Write the file
    wrote = 0
    try:
        with open(model_path, "wb") as f:
            for data in response.iter_content(block_size):
                if should_stop_download_task(model_type, model_filename):
                    break
                wrote = wrote + len(data)
                f.write(data)
    finally:
        remove_download_task(model_type, model_filename)
        if wrote != total_length:
            # Delete the file if it's incomplete
            os.remove(model_path)


class ModelStatus(enum.Enum):
    NotDownloaded = 0
    Downloaded = 1
    Downloading = 2
    Incomplete = 3


def get_model_status(model_url, model_type, model_filename):
    remote_size = get_model_remote_size(model_url)
    local_size = get_model_local_size(model_type, model_filename)

    percentage = 0
    if remote_size and local_size:
        percentage = local_size * 100 / remote_size

    if is_download_task_exists(model_type, model_filename):
        return ModelStatus.Downloading, percentage

    if is_model_file_exists(model_type, model_filename):
        if get_model_remote_size(model_url) == get_model_local_size(
            model_type, model_filename
        ):
            return ModelStatus.Downloaded, percentage
        else:
            return ModelStatus.Incomplete, percentage

    return ModelStatus.NotDownloaded, percentage


def download_model(model_url, model_type, model_filename):
    model_status, _ = get_model_status(model_url, model_type, model_filename)
    if model_status == ModelStatus.Downloaded:
        print("Model has already been downloaded.")
        return gr.Button.update(
            "Already Downloaded", variant="secondary"
        ), gr.Button.update(visible=True)

    if model_status == ModelStatus.Downloading:
        return gr.Button.update(
            "Downloading...", variant="secondary"
        ), gr.Button.update("Cancel", visible=True)

    print(f"Downloading model from {model_url}...")
    # Download the model in a separate thread
    t = threading.Thread(
        target=_download_model_worker,
        args=(model_url, model_type, model_filename),
    )
    t.start()

    return gr.Button.update("Downloading...", variant="secondary"), gr.Button.update(
        "Cancel", visible=True
    )

def delete_model(model_type, model_filename):
    # stop the download task if it's running
    if is_download_task_exists(model_type, model_filename):
        stop_download_task(model_type, model_filename)

    # delete the model file if it exists
    base_dir = "."
    model_path = os.path.join(base_dir, "models", model_type, model_filename)
    if os.path.exists(model_path):
        os.remove(model_path)

    return [gr.Button.update("Download", visible=True), gr.Button.update(visible=False)]


def upload_model(model_files, model_type):
    for file_obj in model_files:
        filename = file_obj.name.split("/")[-1]
        # remove hash after _ from filename but keep the extension
        original_name = filename.split("_")[0] + os.path.splitext(filename)[1]
        # Move file to models folder
        base_dir = "."
        model_path = os.path.join(base_dir, "models", model_type, original_name)

        parent_dir = os.path.dirname(model_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        file_obj.close()
        shutil.move(file_obj.name, model_path)

    return gr.Files.update(None), f"Uploaded {len(model_files)} model(s)."


def is_model_file_exists(model_type, model_filename):
    base_dir = "."

    model_path = os.path.join(base_dir, "models", model_type, model_filename)
    # if the path exists and is a file
    result = os.path.exists(model_path) and os.path.isfile(model_path)

    return result

def get_download_buttons(model_url, model_type, model_filename):
    model_status, percentage = get_model_status(
            model_url, model_type, model_filename
        )

    buttons = [("Download", True), ("Cancel", False)]
    if model_status == ModelStatus.Downloading:
        buttons = [(f"Downloading...({percentage:.1f}%)", True), ("Cancel", True)]
    elif model_status == ModelStatus.Downloaded:
        buttons = [("Already Downloaded", False), ("Delete", True)]
    elif model_status == ModelStatus.Incomplete:
        buttons = [("Incomplete, Redownload", True), ("Delete", True)]

    return buttons


def refresh_models(models, model_index_url):
    if not is_testing and model_index_url.startswith("http"):
        models = requests.get(model_index_url).json()
    else:
        models = json.load(open(model_index_url))

    k = len(models)

    print("refresh models")

    rows_updates = [gr.Row.update(visible=True)] * k
    rows_updates += [gr.Row.update(visible=False)] * (MAX_ROWS - k)

    model_names_updates = [
        gr.HTML.update(f"<pre>{models[i]['name']}</pre>") for i in range(k)
    ] + [gr.HTML.update("")] * (MAX_ROWS - k)
    model_descriptions_updates = [
        gr.Markdown.update(models[i]["description"]) for i in range(k)
    ] + [gr.Markdown.update("")] * (MAX_ROWS - k)

    download_buttons_updates = []
    delete_buttons_updates = []
    for i in range(k):
        model_url = models[i]["url"]
        model_type = models[i]["type"]
        model_filename = models[i]["name"]

        buttons = get_download_buttons(model_url, model_type, model_filename)

        download_buttons_updates.append(
            gr.Button.update(buttons[0][0], visible=buttons[0][1])
        )
        delete_buttons_updates.append(
            gr.Button.update(buttons[1][0], visible=buttons[1][1])
        )

    download_buttons_updates += [gr.Button.update(visible=False)] * (MAX_ROWS - k)
    delete_buttons_updates += [gr.Button.update(visible=False)] * (MAX_ROWS - k)

    model_urls_updates = [models[i]["url"] for i in range(k)] + [""] * (MAX_ROWS - k)
    model_types_updates = [models[i]["type"] for i in range(k)] + [""] * (MAX_ROWS - k)
    model_filenames_updates = [models[i]["name"] for i in range(k)] + [""] * (
        MAX_ROWS - k
    )

    return (
        [models]
        + rows_updates
        + model_names_updates
        + model_descriptions_updates
        + download_buttons_updates
        + delete_buttons_updates
        + model_urls_updates
        + model_types_updates
        + model_filenames_updates
    )

def refresh_manual_model(model_url, model_type, model_filename):
    buttons = get_download_buttons(model_url, model_type, model_filename)

    return (
        gr.Button.update(buttons[0][0], visible=buttons[0][1]),
        gr.Button.update(buttons[1][0], visible=buttons[1][1]),
    )

def add_tab():
    tab_css = """
        #modellist > div {
            padding: 10px;
        }

        #modellist > div:nth-child(odd) {
            background-color: #777;
        }
        #modellist > div:nth-child(even) {
            background-color: #888;
        }
    """

    with gr.Blocks("Model Downloader", css=tab_css) as tab:
        gr.HTML("Download models from the internet and save them to the models folder.")

        models_state = gr.State(PREDEFINED_MODELS)

        with gr.Column():
            with gr.Row().style(equal_height=True):
                with gr.Column(scale=80):
                    model_index_url = gr.Textbox(
                        INITIAL_INDEX_URL, label="Model Index URL"
                    )
                with gr.Column(scale=20):
                    refresh_button = gr.Button("Refresh", variant="primary").style(
                        full_width=True
                    )

        with gr.Box(elem_id="modellist"):
            rows = []
            model_names = []
            model_descriptions = []
            download_buttons = []
            delete_buttons = []
            model_urls = []
            model_types = []
            model_filenames = []

            # Create fixed number of rows, some of which will be hidden
            for row_id in range(MAX_ROWS):
                with gr.Row(visible=False) as row:
                    with gr.Column(scale=30):
                        model_name = gr.HTML()
                    with gr.Column(scale=40):
                        model_description = gr.Markdown()
                    with gr.Column(scale=30):
                        with gr.Row():
                            download_button = gr.Button(
                                "Download",
                                variant="primary",
                                visible=False,
                            )
                            delete_button = gr.Button(
                                "Delete",
                                variant="secondary",
                                visible=False,
                            )

                    model_url = gr.State()
                    model_type = gr.State()
                    model_filename = gr.State()

                    model_names.append(model_name)
                    model_descriptions.append(model_description)
                    download_buttons.append(download_button)
                    delete_buttons.append(delete_button)
                    model_urls.append(model_url)
                    model_types.append(model_type)
                    model_filenames.append(model_filename)

                download_button.click(
                    fn=download_model,
                    inputs=[model_url, model_type, model_filename],
                    outputs=[download_button, delete_button],
                )

                delete_button.click(
                    fn=delete_model,
                    inputs=[model_type, model_filename],
                    outputs=[download_button, delete_button],
                )

                rows.append(row)

        refresh_kwargs = dict(
            fn=refresh_models,
            inputs=[models_state, model_index_url],
            outputs=[models_state]
            + rows
            + model_names
            + model_descriptions
            + download_buttons
            + delete_buttons
            + model_urls
            + model_types
            + model_filenames,
        )

        refresh_button.click(**refresh_kwargs)
        tab.load(**refresh_kwargs)

        with gr.Accordion("Manual Download", open=False):
            # Write a simple interface to download models
            model_url = gr.Text(label="URL", value="https://civitai.com/api/download/models/19575?type=Model&format=SafeTensor")
            model_type = gr.Dropdown(
                choices=MODEL_TYPES,
                label="Type",
                value="Stable-diffusion",
            )
            model_filename = gr.Text(label="Filename", value="ReV-Animated-v1.1.safetensors")
            with gr.Row():
                download_button = gr.Button("Download", variant="primary")
                cancel_button = gr.Button("Cancel", variant="secondary", visible=False)

            download_button.click(
                fn=download_model,
                inputs=[model_url, model_type, model_filename],
                outputs=[download_button, cancel_button],
            )

            cancel_button.click(
                fn=delete_model,
                inputs=[model_type, model_filename],
                outputs=[download_button, cancel_button],
            )

            refresh_button.click(
                fn=refresh_manual_model,
                inputs=[model_url, model_type, model_filename],
                outputs=[download_button, cancel_button],
            )

        with gr.Accordion("Upload Model", open=False):
            # Write a simple interface to upload models
            model_files = gr.Files(label="Model File")
            model_type = gr.Dropdown(
                choices=MODEL_TYPES,
                label="Type",
                value="Stable-diffusion",
            )
            upload_button = gr.Button("Upload")
            model_result = gr.HTML()

            upload_button.click(
                fn=upload_model,
                inputs=[model_files, model_type],
                outputs=[model_files, model_result],
            )

    return [(tab, "Model Downloader", "model_downloader")]


if not is_testing:
    script_callbacks.on_ui_tabs(add_tab)
else:
    tab = add_tab()[0][0]
    tab.launch()
