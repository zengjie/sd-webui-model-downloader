import os
import json

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
    "Lora",
    "deepbooru",
]

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


def download_model(model_url, model_type, model_filename):
    if is_model_file_exists(model_type, model_filename):
        print("Model already downloaded.")
        yield "Already Downloaded"
        return

    print(f"Downloading model from {model_url}...")
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

    yield gr.Button.update("Downloading...", variant="secondary")

    # Write the file
    wrote = 0
    last_percentage = 0
    try:
        with open(model_path, "wb") as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                # output download percentage and size in human readable format
                percentage = int(wrote * 100 / total_length)
                if percentage != last_percentage:
                    humanized_total = convert_bytes(total_length)
                    yield gr.Button.update(
                        f"Downloading... ({percentage}% of {humanized_total})",
                    )
                f.write(data)
    finally:
        if wrote == total_length:
            yield gr.Button.update(f"Downloaded")
        else:
            yield gr.Button.update(f"Download failed. Retry", variant="primary")


def is_model_file_exists(model_type, model_filename):
    base_dir = "."

    model_path = os.path.join(base_dir, "models", model_type, model_filename)
    return os.path.exists(model_path)


def refresh_models(models, model_index_url):
    if not is_testing and model_index_url.startswith("http"):
        models = requests.get(model_index_url).json()
    else:
        models = json.load(open(model_index_url))

    k = len(models)

    rows_updates = [gr.Row.update(visible=True)] * k
    rows_updates += [gr.Row.update(visible=False)] * (MAX_ROWS - k)

    model_names_updates = [
        gr.HTML.update(f"<pre>{models[i]['name']}</pre>") for i in range(k)
    ] + [gr.HTML.update("")] * (MAX_ROWS - k)
    model_descriptions_updates = [
        gr.Markdown.update(models[i]["description"]) for i in range(k)
    ] + [gr.Markdown.update("")] * (MAX_ROWS - k)

    download_buttons_updates = []
    for i in range(k):
        model_type = models[i]["type"]
        model_filename = models[i]["name"]
        if is_model_file_exists(model_type, model_filename):
            download_buttons_updates += [
                gr.Button.update(
                    value="Downloaded",
                    variant="secondary",
                )
            ]
        else:
            download_buttons_updates += [
                gr.Button.update(
                    value="Download",
                    variant="primary",
                )
            ]
    download_buttons_updates += [gr.Button.update(value="")] * (MAX_ROWS - k)

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
        + model_urls_updates
        + model_types_updates
        + model_filenames_updates
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

            with gr.Row():
                gr.CheckboxGroup(
                    ["Downloaded"] + MODEL_TYPES,
                    value=MODEL_TYPES,
                    label="Filters",
                    interactive=True,
                )

        with gr.Box(elem_id="modellist"):
            rows = []
            model_names = []
            model_descriptions = []
            download_buttons = []
            model_urls = []
            model_types = []
            model_filenames = []

            # Create fixed number of rows, some of which will be hidden
            for row_id in range(MAX_ROWS):
                with gr.Row() as row:
                    current_row = gr.State(row_id)
                    try:
                        model = models_state.value[current_row.value]
                    except IndexError:
                        model = {
                            "name": "",
                            "description": "",
                            "url": "",
                            "type": "",
                        }

                    with gr.Column(scale=30):
                        model_name = gr.HTML(f"<pre>{model['name']}</pre>")
                    with gr.Column(scale=50):
                        model_description = gr.Markdown(model["description"])
                    with gr.Column(scale=20):
                        if not is_model_file_exists(model["type"], model["name"]):
                            download_button = gr.Button(
                                "Download",
                                variant="primary",
                            ).style(full_width=True)
                        else:
                            download_button = gr.Button(
                                "Downloaded",
                                variant="secondary",
                            ).style(full_width=True)

                    model_url = gr.State(model["url"])
                    model_type = gr.State(model["type"])
                    model_filename = gr.State(model["name"])

                    model_names.append(model_name)
                    model_descriptions.append(model_description)
                    download_buttons.append(download_button)
                    model_urls.append(model_url)
                    model_types.append(model_type)
                    model_filenames.append(model_filename)

                row.visible = row_id < len(models_state.value)

                download_button.click(
                    fn=download_model,
                    inputs=[model_url, model_type, model_filename],
                    outputs=download_button,
                )

                rows.append(row)

        refresh_button.click(
            fn=refresh_models,
            inputs=[models_state, model_index_url],
            outputs=[models_state]
            + rows
            + model_names
            + model_descriptions
            + download_buttons
            + model_urls
            + model_types
            + model_filenames,
        )

        with gr.Accordion("Manual Download", open=False):
            # Write a simple interface to download models
            model_url = gr.Text(label="URL", value="https://speed.hetzner.de/100MB.bin")
            model_type = gr.Dropdown(
                choices=MODEL_TYPES,
                label="Type",
                value="Stable-diffusion",
            )
            model_filename = gr.Text(label="Filename", value="")
            download_button = gr.Button("Download")
            download_result = gr.Textbox("", label="Output")

            download_button.click(
                fn=download_model,
                inputs=[model_url, model_type, model_filename],
                outputs=download_result,
            )

    return [(tab, "Model Downloader", "model_downloader")]


if not is_testing:
    script_callbacks.on_ui_tabs(add_tab)
else:
    tab = add_tab()[0][0]
    tab.queue().launch()
