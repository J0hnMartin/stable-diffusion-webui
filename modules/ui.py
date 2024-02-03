<<<<<<< HEAD
import os
import json
import mimetypes
from functools import reduce
=======
import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

import gradio as gr
import gradio.utils
import numpy as np
<<<<<<< HEAD
from PIL import Image
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules import timer, shared, theme, sd_models, script_callbacks, modelloader, prompt_parser, ui_common, ui_loadsave, ui_symbols, generation_parameters_copypaste
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.paths import script_path, data_path
from modules.dml import directml_override_opts
import modules.scripts
import modules.textual_inversion.ui
import modules.hypernetworks.ui
import modules.errors

=======
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import gradio_extensons  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, scripts, sd_samplers, processing, ui_extra_networks, ui_toprow
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.generation_parameters_copypaste as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.generation_parameters_copypaste import image_from_url_text

create_setting_component = ui_settings.create_setting_component

warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)
warnings.filterwarnings("default" if opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

modules.errors.install()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
log = shared.log
opts = shared.opts
cmd_opts = shared.cmd_opts
ui_system_tabs = None
switch_values_symbol = ui_symbols.switch
detect_image_size_symbol = ui_symbols.detect
paste_symbol = ui_symbols.paste
clear_prompt_symbol = ui_symbols.clear
restore_progress_symbol = ui_symbols.apply
folder_symbol = ui_symbols.folder
extra_networks_symbol = ui_symbols.networks
apply_style_symbol = ui_symbols.apply
save_style_symbol = ui_symbols.save
txt2img_paste_fields = []
img2img_paste_fields = []
txt2img_args = []
img2img_args = []
paste_function = None


# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type('image/webp', '.webp')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

<<<<<<< HEAD
=======
if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


<<<<<<< HEAD
def create_output_panel(tabname, outdir): # pylint: disable=unused-argument # outdir is used by extensions
    a, b, c, _d, e = ui_common.create_output_panel(tabname)
    return a, b, c, e


def plaintext_to_html(text): # may be referenced by extensions
    return ui_common.plaintext_to_html(text)
=======
sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


plaintext_to_html = ui_common.plaintext_to_html
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def infotext_to_html(text): # may be referenced by extensions
    return ui_common.infotext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
<<<<<<< HEAD
    return generation_parameters_copypaste.image_from_url_text(x[0])


def add_style(name: str, prompt: str, negative_prompt: str):
    from modules import styles
    if name is None:
        return [gr_show() for x in range(4)]
    style = styles.Style(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    shared.prompt_styles.save_styles(shared.opts.styles_dir)
    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler):
    from modules import processing, devices
    if hr_upscaler == "None":
        return "Hires resize: None"
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.init_hr()
    with devices.autocast():
        p.init([""], [0], [0])
    return f"Hires resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)
    if not target_width or not target_height:
        return "Hires resize: no image selected"
    return f"Hires resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]
=======
    return image_from_url_text(x[0])


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def parse_style(styles):
    return styles.split('|')


def process_interrogate(interrogation_function, mode, ii_input_files, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    if mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    if mode == 5:
        if len(ii_input_files) > 0:
            images = [f.name for f in ii_input_files]
        else:
            if not os.path.isdir(ii_input_dir):
                log.error(f"Interrogate: Input directory not found: {ii_input_dir}")
                return [gr.update(), None]
            images = shared.listfiles(ii_input_dir)
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
<<<<<<< HEAD
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8')) # pylint: disable=consider-using-with
    return [gr.update(), None]
=======
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def interrogate(image):
    if image is None:
        log.error("Interrogate: no image selected")
        return gr.update()
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    from modules import deepbooru
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


<<<<<<< HEAD
def create_batch_inputs(tab):
    with gr.Accordion(open=False, label="Batch", elem_id=f"{tab}_batch", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_row_batch"):
            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{tab}_batch_count")
            batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id=f"{tab}_batch_size")
            batch_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_batch_switch_btn", label="Switch dims")
            batch_switch_btn.click(lambda w, h: (h, w), inputs=[batch_count, batch_size], outputs=[batch_count, batch_size], show_progress=False)
    return batch_count, batch_size


def create_seed_inputs(tab, reuse_visible=True):
    with gr.Accordion(open=False, label="Seed", elem_id=f"{tab}_seed_group", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_seed_row", variant="compact"):
            seed = gr.Number(label='Initial seed', value=-1, elem_id=f"{tab}_seed", container=True)
            random_seed = ToolButton(ui_symbols.random, elem_id=f"{tab}_random_seed", label='Random seed')
            reuse_seed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_reuse_seed", label='Reuse seed', visible=reuse_visible)
        with FormRow(elem_id=f"{tab}_subseed_row", variant="compact", visible=shared.backend==shared.Backend.ORIGINAL):
            subseed = gr.Number(label='Variation', value=-1, elem_id=f"{tab}_subseed", container=True)
            random_subseed = ToolButton(ui_symbols.random, elem_id=f"{tab}_random_subseed")
            reuse_subseed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_reuse_subseed", visible=reuse_visible)
            subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{tab}_subseed_strength")
        with FormRow(visible=False):
            seed_resize_from_w = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from width", value=0, elem_id=f"{tab}_seed_resize_from_w")
            seed_resize_from_h = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from height", value=0, elem_id=f"{tab}_seed_resize_from_h")
        random_seed.click(fn=lambda: [-1, -1], show_progress=False, inputs=[], outputs=[seed, subseed])
        random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


def create_advanced_inputs(tab):
    with gr.Accordion(open=False, label="Advanced", elem_id=f"{tab}_advanced", elem_classes=["small-accordion"]):
        with gr.Group():
            with FormRow():
                cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='CFG scale', value=4.0, elem_id=f"{tab}_cfg_scale")
                clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=14, step=1, elem_id=f"{tab}_clip_skip", interactive=True)
            with FormRow():
                image_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Secondary CFG scale', value=4.0, elem_id=f"{tab}_image_cfg_scale")
                diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.7, elem_id=f"{tab}_image_cfg_rescale", visible=shared.backend == shared.Backend.DIFFUSERS)
        with gr.Group():
            with FormRow():
                full_quality = gr.Checkbox(label='Full quality', value=True, elem_id=f"{tab}_full_quality")
                restore_faces = gr.Checkbox(label='Face restore', value=False, visible=len(shared.face_restorers) > 1, elem_id=f"{tab}_restore_faces")
                tiling = gr.Checkbox(label='Tiling', value=False, elem_id=f"{tab}_tiling", visible=shared.backend == shared.Backend.ORIGINAL)
        with gr.Group(visible=shared.backend == shared.Backend.DIFFUSERS):
            with FormRow():
                hdr_clamp = gr.Checkbox(label='HDR clamp', value=False, elem_id=f"{tab}_hdr_clamp")
                hdr_boundary = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=4.0,  label='Range', elem_id=f"{tab}_hdr_boundary")
                hdr_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.95,  label='Threshold', elem_id=f"{tab}_hdr_threshold")
            with FormRow():
                hdr_center = gr.Checkbox(label='HDR center', value=False, elem_id=f"{tab}_hdr_center")
                hdr_channel_shift = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0,  label='Channel shift', elem_id=f"{tab}_hdr_channel_shift")
                hdr_full_shift = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1,  label='Full shift', elem_id=f"{tab}_hdr_full_shift")
            with FormRow():
                hdr_maximize = gr.Checkbox(label='HDR maximize', value=False, elem_id=f"{tab}_hdr_maximize")
                hdr_max_center = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.6,  label='Center', elem_id=f"{tab}_hdr_max_center")
                hdr_max_boundry = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0,  label='Range', elem_id=f"{tab}_hdr_max_boundry")
    return cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry


def create_resize_inputs(tab, images, time_selector=False, scale_visible=True, mode=None):
    dummy_component = gr.Number(visible=False, value=0)
    with gr.Accordion(open=False, label="Resize", elem_classes=["small-accordion"], elem_id=f"{tab}_resize_group"):
        with gr.Row():
            if mode is not None:
                resize_mode = gr.Radio(label="Resize mode", elem_id=f"{tab}_resize_mode", choices=shared.resize_modes, type="index", value=mode, visible=False)
            else:
                resize_mode = gr.Radio(label="Resize mode", elem_id=f"{tab}_resize_mode", choices=shared.resize_modes, type="index", value='None')
            resize_time = gr.Radio(label="Resize order", elem_id=f"{tab}_resize_order", choices=['Before', 'After'], value="Before", visible=time_selector)
        with gr.Row():
            resize_name = gr.Dropdown(label="Resize method", elem_id=f"{tab}_resize_name", choices=[x.name for x in shared.sd_upscalers], value=opts.upscaler_for_img2img)
            create_refresh_button(resize_name, modelloader.load_upscalers, lambda: {"choices": modelloader.load_upscalers()}, 'refresh_upscalers')

        with FormRow(visible=True) as _resize_group:
            with gr.Column(elem_id=f"{tab}_column_size"):
                selected_scale_tab = gr.State(value=0) # pylint: disable=abstract-class-instantiated
                with gr.Tabs():
                    with gr.Tab(label="Resize to") as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id=f"{tab}_column_size"):
                                with FormRow():
                                    width = gr.Slider(minimum=64, maximum=8192, step=8, label="Width", value=512, elem_id=f"{tab}_width")
                                    height = gr.Slider(minimum=64, maximum=8192, step=8, label="Height", value=512, elem_id=f"{tab}_height")
                                    res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_res_switch_btn")
                                    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
                                    detect_image_size_btn = ToolButton(value=ui_symbols.detect, elem_id=f"{tab}_detect_image_size_btn")
                                    detect_image_size_btn.click(fn=lambda w, h, _: (w or gr.update(), h or gr.update()), _js="currentImg2imgSourceResolution", inputs=[dummy_component, dummy_component, dummy_component], outputs=[width, height], show_progress=False)

                    with gr.Tab(label="Resize by") as tab_scale_by:
                        scale_by = gr.Slider(minimum=0.05, maximum=8.0, step=0.05, label="Scale", value=1.0, elem_id=f"{tab}_scale")
                        if scale_visible:
                            with FormRow():
                                scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id=f"{tab}_scale_resolution_preview")
                                gr.Slider(label="Unused", elem_id=f"{tab}_unused_scale_by_slider")
                                button_update_resize_to = gr.Button(visible=False, elem_id=f"{tab}_update_resize_to")

                            on_change_args = dict(fn=resize_from_to_html, _js="currentImg2imgSourceResolution", inputs=[dummy_component, dummy_component, scale_by], outputs=scale_by_html, show_progress=False)
                            scale_by.release(**on_change_args)
                            button_update_resize_to.click(**on_change_args)

                    for component in images:
                        component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])
            # resize_mode.change(fn=lambda x: gr.update(visible=x != 0), inputs=[resize_mode], outputs=[_resize_group])
    return resize_mode, resize_name, width, height, scale_by, selected_scale_tab, resize_time


def connect_clear_prompt(button): # pylint: disable=unused-argument
    pass


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index: int):
        res = -1
        try:
            gen_info = json.loads(gen_info_string)
            log.debug(f'Reuse: info={gen_info}')
            index -= gen_info.get('index_of_first_image', 0)
            index = int(index)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]
        except json.decoder.JSONDecodeError:
            if gen_info_string != '':
                log.error(f"Error parsing JSON generation info: {gen_info_string}")
        return [res, gr_show(False)]

    dummy_component = gr.Number(visible=False, value=0)
    reuse_seed.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress=False, inputs=[generation_info, dummy_component], outputs=[seed, dummy_component])


def update_token_counter(text, steps):
    from modules import extra_networks, sd_hijack
    try:
        text, _ = extra_networks.parse_prompt(text)
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
=======
def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def update_token_counter(text, steps, *, is_positive=True):
    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
    except Exception:
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    if shared.backend == shared.Backend.ORIGINAL:
        token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    elif shared.backend == shared.Backend.DIFFUSERS:
        if shared.sd_model is not None and hasattr(shared.sd_model, 'tokenizer'):
            tokenizer = shared.sd_model.tokenizer
            if tokenizer is None:
                token_count = 0
                max_length = 75
            else:
                has_bos_token = tokenizer.bos_token_id is not None
                has_eos_token = tokenizer.eos_token_id is not None
                ids = [shared.sd_model.tokenizer(prompt) for prompt in prompts]
                if len(ids) > 0 and hasattr(ids[0], 'input_ids'):
                    ids = [x.input_ids for x in ids]
                token_count = max([len(x) for x in ids]) - int(has_bos_token) - int(has_eos_token)
                max_length = tokenizer.model_max_length - int(has_bos_token) - int(has_eos_token)
        else:
            token_count = 0
            max_length = 75
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


<<<<<<< HEAD
def create_toprow(is_img2img: bool = False, id_part: str = None):
    if id_part is None:
        id_part = "img2img" if is_img2img else "txt2img"
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(elem_id=f"{id_part}_prompt", label="Prompt", show_label=False, lines=3, placeholder="Prompt", elem_classes=["prompt"])
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(elem_id=f"{id_part}_neg_prompt", label="Negative prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt"])
        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id=f"{id_part}_interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id=f"{id_part}_deepbooru")
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
            with gr.Row(elem_id=f"{id_part}_generate_line2"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt")
                interrupt.click(fn=lambda: shared.state.interrupt(), _js="requestInterrupt", inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=lambda: shared.state.pause(), _js='checkPaused', inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                button_paste = gr.Button(value='Restore', variant='secondary', elem_id=f"{id_part}_paste") # symbols.paste
                button_clear = gr.Button(value='Clear', variant='secondary', elem_id=f"{id_part}_clear_prompt_btn") # symbols.clear
                button_extra = gr.Button(value='Networks', variant='secondary', elem_id=f"{id_part}_extra_networks_btn") # symbols.networks
                button_clear.click(fn=lambda *x: ['', ''], inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt], show_progress=False)
            with gr.Row(elem_id=f"{id_part}_counters"):
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[style.name for style in shared.prompt_styles.styles.values()], value=[], multiselect=True)
                _styles_btn_refresh = create_refresh_button(styles, shared.prompt_styles.reload, lambda: {"choices": list(shared.prompt_styles.styles)}, f"{id_part}_styles_refresh")
                # styles_btn_refresh = ToolButton(symbols.refresh, elem_id=f"{id_part}_styles_refresh", visible=True)
                # styles_btn_refresh.click(fn=lambda: gr.update(choices=[style.name for style in shared.prompt_styles.styles.values()]), inputs=[], outputs=[styles])
                styles_btn_select = gr.Button('Select', elem_id=f"{id_part}_styles_select", visible=False)
                styles_btn_select.click(_js="applyStyles", fn=parse_style, inputs=[styles], outputs=[styles])
                styles_btn_apply = ToolButton(ui_symbols.apply, elem_id=f"{id_part}_extra_apply", visible=False)
                styles_btn_apply.click(fn=apply_styles, inputs=[prompt, negative_prompt, styles], outputs=[prompt, negative_prompt, styles])
    return prompt, styles, negative_prompt, submit, button_interrogate, button_deepbooru, button_paste, button_extra, token_counter, token_button, negative_token_counter, negative_token_button
=======
def update_negative_prompt_token_counter(text, steps):
    return update_token_counter(text, steps, is_positive=False)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if shared.cmd_opts.freeze:
        return gr.update()
    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()
    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)
        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()
    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return gr.update()
    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()
    opts.save(shared.config_filename)
    return getattr(opts, key)


<<<<<<< HEAD
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    return ui_common.create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)


def create_sampler_and_steps_selection(choices, tabname):
    def set_sampler_original_options(sampler_options, sampler_algo):
        opts.data['schedulers_brownian_noise'] = 'brownian noise' in sampler_options
        opts.data['schedulers_discard_penultimate'] = 'discard penultimate sigma' in sampler_options
        opts.data['schedulers_sigma'] = sampler_algo
        opts.save(shared.config_filename, silent=True)

    def set_sampler_diffuser_options(sampler_options):
        opts.data['schedulers_use_karras'] = 'karras' in sampler_options
        opts.data['schedulers_use_thresholding'] = 'dynamic thresholding' in sampler_options
        opts.data['schedulers_use_loworder'] = 'low order' in sampler_options
        opts.data['schedulers_rescale_betas'] = 'rescale beta' in sampler_options
        opts.save(shared.config_filename, silent=True)

    with FormRow(elem_classes=['flex-break']):
        sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value='Default', type="index")
        steps = gr.Slider(minimum=1, maximum=99, step=1, label="Sampling steps", elem_id=f"{tabname}_steps", value=20)
    if shared.backend == shared.Backend.ORIGINAL:
        with FormRow(elem_classes=['flex-break']):
            choices = ['brownian noise', 'discard penultimate sigma']
            values = []
            values += ['brownian noise'] if opts.data.get('schedulers_brownian_noise', False) else []
            values += ['discard penultimate sigma'] if opts.data.get('schedulers_discard_penultimate', True) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        with FormRow(elem_classes=['flex-break']):
            opts.data['schedulers_sigma'] = opts.data.get('schedulers_sigma', 'default')
            sampler_algo = gr.Radio(label='Sigma algorithm', choices=['default', 'karras', 'exponential', 'polyexponential'], value=opts.data['schedulers_sigma'], type='value')
        sampler_options.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
        sampler_algo.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
    else:
        with FormRow(elem_classes=['flex-break']):
            choices = ['karras', 'dynamic threshold', 'low order', 'rescale beta']
            values = []
            values += ['karras'] if opts.data.get('schedulers_use_karras', True) else []
            values += ['dynamic threshold'] if opts.data.get('schedulers_use_thresholding', False) else []
            values += ['low order'] if opts.data.get('schedulers_use_loworder', True) else []
            values += ['rescale beta'] if opts.data.get('schedulers_rescale_betas', False) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        sampler_options.change(fn=set_sampler_diffuser_options, inputs=[sampler_options], outputs=[])
    return steps, sampler_index


def create_sampler_inputs(tab):
    from modules import sd_samplers
    with gr.Accordion(open=False, label="Sampler", elem_id=f"{tab}_sampler", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_row_sampler"):
            sd_samplers.set_samplers()
            steps, sampler_index = create_sampler_and_steps_selection(sd_samplers.samplers, tab)
    return steps, sampler_index


def create_hires_inputs(tab):
    with gr.Accordion(open=False, label="Second pass", elem_id=f"{tab}_second_pass", elem_classes=["small-accordion"]):
        with FormGroup():
            with FormRow(elem_id=f"{tab}_hires_row1"):
                enable_hr = gr.Checkbox(label='Enable second pass', value=False, elem_id=f"{tab}_enable_hr")
            with FormRow(elem_id=f"{tab}_hires_row2"):
                latent_index = gr.Dropdown(label='Secondary sampler', elem_id=f"{tab}_sampling_alt", choices=[x.name for x in modules.sd_samplers.samplers], value='Default', type="index")
                denoising_strength = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, label='Denoising strength', value=0.5, elem_id=f"{tab}_denoising_strength")
            with FormRow(elem_id=f"{tab}_hires_finalres", variant="compact"):
                hr_final_resolution = FormHTML(value="", elem_id=f"{tab}_hr_finalres", label="Upscaled resolution", interactive=False)
            with FormRow(elem_id=f"{tab}_hires_fix_row1", variant="compact"):
                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{tab}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                hr_force = gr.Checkbox(label='Force Hires', value=False, elem_id=f"{tab}_hr_force")
            with FormRow(elem_id=f"{tab}_hires_fix_row2", variant="compact"):
                hr_second_pass_steps = gr.Slider(minimum=0, maximum=99, step=1, label='Hires steps', elem_id=f"{tab}_steps_alt", value=20)
                hr_scale = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{tab}_hr_scale")
            with FormRow(elem_id=f"{tab}_hires_fix_row3", variant="compact"):
                hr_resize_x = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize width to", value=0, elem_id=f"{tab}_hr_resize_x")
                hr_resize_y = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize height to", value=0, elem_id=f"{tab}_hr_resize_y")
        with FormGroup(visible=shared.backend == shared.Backend.DIFFUSERS):
            with FormRow(elem_id=f"{tab}_refiner_row1", variant="compact"):
                refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Refiner start', value=0.8, elem_id=f"{tab}_refiner_start")
                refiner_steps = gr.Slider(minimum=0, maximum=99, step=1, label="Refiner steps", elem_id=f"{tab}_refiner_steps", value=5)
            with FormRow(elem_id=f"{tab}_refiner_row3", variant="compact"):
                refiner_prompt = gr.Textbox(value='', label='Secondary prompt', elem_id=f"{tab}_refiner_prompt")
            with FormRow(elem_id="txt2img_refiner_row4", variant="compact"):
                refiner_negative = gr.Textbox(value='', label='Secondary negative prompt', elem_id=f"{tab}_refiner_neg_prompt")
    return enable_hr, latent_index, denoising_strength, hr_final_resolution, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative


def get_value_for_setting(key):
    value = getattr(opts, key)
    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}
    return gr.update(value=value, **args)


def ordered_ui_categories():
    return ['dimensions', 'sampler', 'seed', 'denoising', 'cfg', 'checkboxes', 'accordions', 'override_settings', 'scripts'] # a1111 compatibility item, not implemented
=======
def create_output_panel(tabname, outdir, toprow=None):
    return ui_common.create_output_panel(tabname, outdir, toprow)


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_name = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])

    return steps, sampler_name


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def create_override_inputs(tab): # pylint: disable=unused-argument
    with FormRow(elem_id=f"{tab}_override_settings_row"):
        override_settings = gr.Dropdown([], value=None, label="Override settings", visible=False, elem_id=f"{tab}_override_settings", multiselect=True)
        override_settings.change(fn=lambda x: gr.Dropdown.update(visible=len(x) > 0), inputs=[override_settings], outputs=[override_settings])
    return override_settings


def create_ui(startup_timer = None):
    if startup_timer is None:
        timer.startup = timer.Timer()
    reload_javascript()
    generation_parameters_copypaste.reset()

<<<<<<< HEAD
    import modules.txt2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, txt2img_submit, _interrogate, _deepbooru, txt2img_paste, txt2img_extra_networks_button, txt2img_token_counter, txt2img_token_button, txt2img_negative_token_counter, txt2img_negative_token_button = create_toprow(is_img2img=False, id_part="txt2img")

        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)
        txt_prompt_img.change(fn=modules.images.image_data, inputs=[txt_prompt_img], outputs=[txt2img_prompt, txt_prompt_img])

        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, txt2img_extra_networks_button, 'txt2img', skip_indexing=opts.extra_network_skip_indexing)
            timer.startup.record('ui-extra-networks')

        with gr.Row(elem_id="txt2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
=======
    parameters_copypaste.reset()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)

        dummy_component = gr.Label(visible=False)

        extra_tabs = gr.Tabs(elem_id="txt2img_extra_tabs")
        extra_tabs.__enter__()

        with gr.Tab("Generation", id="txt2img_generation") as txt2img_generation_tab, ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.txt2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="txt2img_settings"))

                scripts.scripts_txt2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), "txt2img")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

                with FormRow():
                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="txt2img_width")
                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="txt2img_height")
                    res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id="txt2img_res_switch_btn", label="Switch dims")
                    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

<<<<<<< HEAD
                with FormGroup(elem_classes="settings-accordion"):
=======
                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", tooltip="Switch width/height")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

                    steps, sampler_index = create_sampler_inputs('txt2img')
                    batch_count, batch_size = create_batch_inputs('txt2img')
                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('txt2img')
                    cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry = create_advanced_inputs('txt2img')
                    enable_hr, latent_index, denoising_strength, hr_final_resolution, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative = create_hires_inputs('txt2img')
                    override_settings = create_override_inputs('txt2img')

<<<<<<< HEAD
                txt2img_script_inputs = modules.scripts.scripts_txt2img.setup_ui()

            hr_resolution_preview_inputs = [width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler]
            for preview_input in hr_resolution_preview_inputs:
                preview_input.change(
=======
                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                            with InputAccordion(False, label="Hires. fix", elem_id="txt2img_hr") as enable_hr:
                                with enable_hr.extra():
                                    hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                    hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                    hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                                with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                    hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                    hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                    hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                                with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as hr_sampler_container:

                                    hr_checkpoint_name = gr.Dropdown(label='Hires checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint")
                                    create_refresh_button(hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                    hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")

                                with FormRow(elem_id="txt2img_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as hr_prompts_container:
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                    with gr.Column(scale=80):
                                        with gr.Row():
                                            hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                            scripts.scripts_txt2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = scripts.scripts_txt2img.setup_ui()

                    if category not in {"accordions"}:
                        scripts.scripts_txt2img.setup_ui_for_section(category)

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

            for component in hr_resolution_preview_inputs:
                event = component.release if isinstance(component, gr.Slider) else component.change

                event(
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                    fn=calc_resolution_hires,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
<<<<<<< HEAD

            txt2img_gallery, txt2img_generation_info, txt2img_html_info, _txt2img_html_info_formatted, txt2img_html_log = ui_common.create_output_panel("txt2img")
            connect_reuse_seed(seed, reuse_seed, txt2img_generation_info, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, txt2img_generation_info, is_subseed=True)

            global txt2img_args # pylint: disable=global-statement
            dummy_component = gr.Textbox(visible=False, value='dummy')
            txt2img_args = [
                dummy_component,
                txt2img_prompt, txt2img_negative_prompt, txt2img_prompt_styles,
                steps, sampler_index, latent_index,
                full_quality, restore_faces, tiling,
                batch_count, batch_size,
                cfg_scale, image_cfg_scale, diffusers_guidance_rescale,
                clip_skip,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                height, width,
                enable_hr, denoising_strength,
                hr_scale, hr_upscaler, hr_force, hr_second_pass_steps, hr_resize_x, hr_resize_y,
                refiner_steps, refiner_start, refiner_prompt, refiner_negative,
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
                override_settings,
            ]
            txt2img_dict = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit_txt2img",
                inputs=txt2img_args + txt2img_script_inputs,
=======
                event(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples, toprow)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    toprow.prompt,
                    toprow.negative_prompt,
                    toprow.ui_styles.dropdown,
                    steps,
                    sampler_name,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
                    hr_checkpoint_name,
                    hr_sampler_name,
                    hr_prompt,
                    hr_negative_prompt,
                    override_settings,

                ] + custom_inputs,

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                outputs=[
                    txt2img_gallery,
                    txt2img_generation_info,
                    txt2img_html_info,
                    txt2img_html_log,
                ],
                show_progress=False,
            )
            txt2img_prompt.submit(**txt2img_dict)
            txt2img_submit.click(**txt2img_dict)

<<<<<<< HEAD
            global txt2img_paste_fields # pylint: disable=global-statement
            txt2img_paste_fields = [
                # prompt
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                # main
                (width, "Size-1"),
                (height, "Size-2"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # advanced
                (cfg_scale, "CFG scale"),
                (clip_skip, "Clip skip"),
                (image_cfg_scale, "Image CFG scale"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (full_quality, "Full quality"),
                (restore_faces, "Face restoration"),
                (tiling, "Tiling"),
                # second pass
                (enable_hr, "Second pass"),
                (latent_index, "Latent sampler"),
                (denoising_strength, "Denoising strength"),
=======
            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressTxt2img",
                inputs=[dummy_component],
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            txt2img_paste_fields = [
                (toprow.prompt, "Prompt"),
                (toprow.negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_name, "Sampler"),
                (cfg_scale, "CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d)),
                (hr_scale, "Hires upscale"),
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                (hr_upscaler, "Hires upscaler"),
                (hr_force, "Hires force"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_scale, "Hires upscale"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
<<<<<<< HEAD
                # refiner
                (refiner_start, "Refiner start"),
                (refiner_steps, "Refiner steps"),
                (refiner_prompt, "Prompt2"),
                (refiner_negative, "Negative2"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            txt2img_bindings = generation_parameters_copypaste.ParamBinding(paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(txt2img_bindings)

            txt2img_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[txt2img_token_counter])
            txt2img_negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[txt2img_negative_token_counter])
=======
                (hr_checkpoint_name, "Hires checkpoint"),
                (hr_sampler_name, "Hires sampler"),
                (hr_sampler_container, lambda d: gr.update(visible=True) if d.get("Hires sampler", "Use same sampler") != "Use same sampler" or d.get("Hires checkpoint", "Use same checkpoint") != "Use same checkpoint" else gr.update()),
                (hr_prompt, "Hires prompt"),
                (hr_negative_prompt, "Hires negative prompt"),
                (hr_prompts_container, lambda d: gr.update(visible=True) if d.get("Hires prompt", "") != "" or d.get("Hires negative prompt", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="txt2img", source_text_component=toprow.prompt, source_image_component=None,
            ))

            txt2img_preview_params = [
                toprow.prompt,
                toprow.negative_prompt,
                steps,
                sampler_name,
                cfg_scale,
                scripts.scripts_txt2img.script('Seed').seed,
                width,
                height,
            ]

            toprow.token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps], outputs=[toprow.negative_token_counter])
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        extra_networks_ui = ui_extra_networks.create_ui(txt2img_interface, [txt2img_generation_tab], 'txt2img')
        ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)

<<<<<<< HEAD
    timer.startup.record("ui-txt2img")

    import modules.img2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_interrogate, img2img_deepbooru, img2img_paste, img2img_extra_networks_button, img2img_token_counter, img2img_token_button, img2img_negative_token_counter, img2img_negative_token_button = create_toprow(is_img2img=True, id_part="img2img")
        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, img2img_extra_networks_button, 'img2img', skip_indexing=opts.extra_network_skip_indexing)

        with FormRow(elem_id="img2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
=======
        extra_tabs.__exit__()

    scripts.scripts_current = scripts.scripts_img2img
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=True, is_compact=shared.opts.compact_prompt_box)

        extra_tabs = gr.Tabs(elem_id="img2img_extra_tabs")
        extra_tabs.__enter__()

        with gr.Tab("Generation", id="img2img_generation") as img2img_generation_tab, ResizeHandleRow(equal_height=False):
            with ExitStack() as stack:
                if shared.opts.img2img_settings_accordion:
                    stack.enter_context(gr.Accordion("Open for Settings", open=False))
                stack.enter_context(gr.Column(variant='compact', elem_id="img2img_settings"))

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                copy_image_buttons = []
                copy_image_destinations = {}

                def copy_image(img):
                    return img['image'] if isinstance(img, dict) and 'image' in img else img

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        for title, name in zip(['➠ Image', '➠ Sketch', '➠ Inpaint', '➠ Inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue
                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

<<<<<<< HEAD
                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0) # pylint: disable=abstract-class-instantiated
                    with gr.TabItem('Image', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=512)
                        add_copy_image_controls('img2img', init_img)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        inpaint_color_sketch_orig = gr.State(None) # pylint: disable=abstract-class-instantiated
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state
                            return state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Upload images or process images in a directory" +
                            "<br>Add inpaint batch mask directory to enable inpaint batch processing"
                            f"{hidden}</p>"
                        )
                        img2img_batch_files = gr.Files(label="Batch Process", interactive=True, elem_id="img2img_image_batch")
                        img2img_batch_input_dir = gr.Textbox(label="Inpaint batch input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Inpaint batch output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]
                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                for button, name, elem in copy_image_buttons:
                    button.click(fn=copy_image, inputs=[elem], outputs=[copy_image_destinations[name]])
                    button.click(fn=lambda: None, _js=f"switch_to_{name.replace(' ', '_')}", inputs=[], outputs=[])

                with FormGroup(elem_classes="settings-accordion"):

                    steps, sampler_index = create_sampler_inputs('img2img')
                    resize_mode, resize_name, width, height, scale_by, selected_scale_tab, _resize_time = create_resize_inputs('img2img', [init_img, sketch])
                    batch_count, batch_size = create_batch_inputs('img2img')
                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('img2img')
=======
                scripts.scripts_img2img.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "prompt":
                        toprow.create_inline_toprow_prompts()

                    if category == "image":
                        with gr.Tabs(elem_id="mode_img2img"):
                            img2img_selected_tab = gr.State(0)

                            with gr.TabItem('img2img', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                                init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=opts.img2img_editor_height)
                                add_copy_image_controls('img2img', init_img)

                            with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                                sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_sketch_default_brush_color)
                                add_copy_image_controls('sketch', sketch)

                            with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                                init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_mask_brush_color)
                                add_copy_image_controls('inpaint', init_img_with_mask)

                            with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                                inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_sketch_default_brush_color)
                                inpaint_color_sketch_orig = gr.State(None)
                                add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                                def update_orig(image, state):
                                    if image is not None:
                                        same_size = state is not None and state.size == image.size
                                        has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                        edited = same_size and has_exact_match
                                        return image if not edited or state is None else state

                                inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                            with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                                init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                                init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id="img_inpaint_mask")

                            with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                                hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                                gr.HTML(
                                    "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running." +
                                    "<br>Use an empty output directory to save pictures normally instead of writing to the output directory." +
                                    f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
                                    f"{hidden}</p>"
                                )
                                img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                                img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                                img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory (required for inpaint batch processing only)", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")
                                with gr.Accordion("PNG info", open=False):
                                    img2img_batch_use_png_info = gr.Checkbox(label="Append png info to prompts", **shared.hide_dirs, elem_id="img2img_batch_use_png_info")
                                    img2img_batch_png_info_dir = gr.Textbox(label="PNG info directory", **shared.hide_dirs, placeholder="Leave empty to use input directory", elem_id="img2img_batch_png_info_dir")
                                    img2img_batch_png_info_props = gr.CheckboxGroup(["Prompt", "Negative prompt", "Seed", "CFG scale", "Sampler", "Steps", "Model hash"], label="Parameters to take from png info", info="Prompts from png info will be appended to prompts set in ui.")

                            img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                            for i, tab in enumerate(img2img_tabs):
                                tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                        def copy_image(img):
                            if isinstance(img, dict) and 'image' in img:
                                return img['image']

                            return img

                        for button, name, elem in copy_image_buttons:
                            button.click(
                                fn=copy_image,
                                inputs=[elem],
                                outputs=[copy_image_destinations[name]],
                            )
                            button.click(
                                fn=lambda: None,
                                _js=f"switch_to_{name.replace(' ', '_')}",
                                inputs=[],
                                outputs=[],
                            )

                        with FormRow():
                            resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), "img2img")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

                    with gr.Accordion(open=False, label="Denoise", elem_classes=["small-accordion"], elem_id="img2img_denoise_group"):
                        with FormRow():
<<<<<<< HEAD
                            denoising_strength = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, label='Denoising strength', value=0.50, elem_id="img2img_denoising_strength")
                            refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="img2img_refiner_start")

                    cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry = create_advanced_inputs('img2img')

                    with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                        with FormRow():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")
                        with FormRow():
                            with gr.Column():
=======
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.State(value=0)

                                with gr.Tabs():
                                    with gr.Tab(label="Resize to", elem_id="img2img_tab_resize_to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="img2img_width")
                                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="img2img_height")
                                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="img2img_res_switch_btn", tooltip="Switch width/height")
                                                detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id="img2img_detect_image_size_btn", tooltip="Auto detect size from img2img")

                                    with gr.Tab(label="Resize by", elem_id="img2img_tab_resize_by") as tab_scale_by:
                                        scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.05, label="Scale", value=1.0, elem_id="img2img_scale")

                                        with FormRow():
                                            scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id="img2img_scale_resolution_preview")
                                            gr.Slider(label="Unused", elem_id="img2img_unused_scale_by_slider")
                                            button_update_resize_to = gr.Button(visible=False, elem_id="img2img_update_resize_to")

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        _js="currentImg2imgSourceResolution",
                                        inputs=[dummy_component, dummy_component, scale_by],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="img2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "denoising":
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75, elem_id="img2img_denoising_strength")

                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="img2img_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id="img2img_image_cfg_scale", visible=False)

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id="img2img_accordions", elem_classes="accordions"):
                            scripts.scripts_img2img.setup_ui_for_section(category)

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="img2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('img2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = scripts.scripts_img2img.setup_ui()

                    elif category == "inpaint":
                        with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                                mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")

                            with FormRow():
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                                inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")
                            with gr.Column():
                                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'noise', 'nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")
                        with FormRow():
                            with gr.Column():
                                inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")
                            with gr.Column():
                                inpaint_full_res_padding = gr.Slider(label='Masked padding', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                        def select_img2img_tab(tab):
                            return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3)

                        for i, elem in enumerate(img2img_tabs):
                            elem.select(fn=lambda tab=i: select_img2img_tab(tab), inputs=[], outputs=[inpaint_controls, mask_alpha]) # pylint: disable=cell-var-from-loop

                override_settings = create_override_inputs('img2img')

<<<<<<< HEAD
                with FormGroup(elem_id="img2img_script_container"):
                    img2img_script_inputs = modules.scripts.scripts_img2img.setup_ui()

            img2img_gallery, img2img_generation_info, img2img_html_info, _img2img_html_info_formatted, img2img_html_log = ui_common.create_output_panel("img2img")

            connect_reuse_seed(seed, reuse_seed, img2img_generation_info, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, img2img_generation_info, is_subseed=True)

            img2img_prompt_img.change(fn=modules.images.image_data, inputs=[img2img_prompt_img], outputs=[img2img_prompt, img2img_prompt_img])
            dummy_component1 = gr.Textbox(visible=False, value='dummy')
            dummy_component2 = gr.Number(visible=False, value=0)
            global img2img_args # pylint: disable=global-statement
            img2img_args = [
                dummy_component1, dummy_component2,
                img2img_prompt, img2img_negative_prompt, img2img_prompt_styles,
                init_img,
                sketch,
                init_img_with_mask,
                inpaint_color_sketch,
                inpaint_color_sketch_orig,
                init_img_inpaint,
                init_mask_inpaint,
                steps,
                sampler_index, latent_index,
                mask_blur, mask_alpha,
                inpainting_fill,
                full_quality, restore_faces, tiling,
                batch_count, batch_size,
                cfg_scale, image_cfg_scale,
                diffusers_guidance_rescale,
                refiner_steps,
                refiner_start,
                clip_skip,
                denoising_strength,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                selected_scale_tab,
                height, width,
                scale_by,
                resize_mode, resize_name,
                inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
                override_settings,
            ]
            img2img_dict = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs= img2img_args + img2img_script_inputs,
=======
                    if category not in {"accordions"}:
                        scripts.scripts_img2img.setup_ui_for_section(category)

            # the code below is meant to update the resolution label after the image in the image selection UI has changed.
            # as it is now the event keeps firing continuously for inpaint edits, which ruins the page with constant requests.
            # I assume this must be a gradio bug and for now we'll just do it for non-inpaint inputs.
            for component in [init_img, sketch]:
                component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

            def select_img2img_tab(tab):
                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

            for i, elem in enumerate(img2img_tabs):
                elem.select(
                    fn=lambda tab=i: select_img2img_tab(tab),
                    inputs=[],
                    outputs=[inpaint_controls, mask_alpha],
                )

            img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples, toprow)

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    dummy_component,
                    toprow.prompt,
                    toprow.negative_prompt,
                    toprow.ui_styles.dropdown,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    steps,
                    sampler_name,
                    mask_blur,
                    mask_alpha,
                    inpainting_fill,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    selected_scale_tab,
                    height,
                    width,
                    scale_by,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    img2img_batch_inpaint_mask_dir,
                    override_settings,
                    img2img_batch_use_png_info,
                    img2img_batch_png_info_props,
                    img2img_batch_png_info_dir,
                ] + custom_inputs,
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                outputs=[
                    img2img_gallery,
                    img2img_generation_info,
                    img2img_html_info,
                    img2img_html_log,
                ],
                show_progress=False,
            )
            img2img_prompt.submit(**img2img_dict)
            submit.click(**img2img_dict)
            dummy_component = gr.Textbox(visible=False, value='dummy')

            interrogate_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_files,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    init_img_inpaint,
                ],
                outputs=[toprow.prompt, dummy_component],
            )
            img2img_interrogate.click(fn=lambda *args: process_interrogate(interrogate, *args), **interrogate_args)
            img2img_deepbooru.click(fn=lambda *args: process_interrogate(interrogate_deepbooru, *args), **interrogate_args)

<<<<<<< HEAD
            img2img_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_prompt, steps], outputs=[img2img_token_counter])
            img2img_negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_negative_prompt, steps], outputs=[img2img_negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)
            global img2img_paste_fields # pylint: disable=global-statement
            img2img_paste_fields = [
                # prompt
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # resize
                (resize_mode, "Resize mode"),
                (width, "Size-1"),
                (height, "Size-2"),
                (scale_by, "Resize scale"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # denoise
                (denoising_strength, "Denoising strength"),
                (refiner_start, "Refiner start"),
                # advanced
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (clip_skip, "Clip skip"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (full_quality, "Full quality"),
                (restore_faces, "Face restoration"),
                (tiling, "Tiling"),
                # inpaint
                (mask_blur, "Mask blur"),
                (mask_alpha, "Mask alpha"),
                (inpainting_mask_invert, "Mask invert"),
                (inpainting_fill, "Masked content"),
                (inpaint_full_res, "Mask area"),
                (inpaint_full_res_padding, "Masked padding"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            generation_parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            img2img_bindings = generation_parameters_copypaste.ParamBinding(paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(img2img_bindings)

    timer.startup.record("ui-img2img")
=======
            toprow.prompt.submit(**img2img_args)
            toprow.submit.click(**img2img_args)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('img2img')}", inputs=None, outputs=None, show_progress=False)

            detect_image_size_btn.click(
                fn=lambda w, h, _: (w or gr.update(), h or gr.update()),
                _js="currentImg2imgSourceResolution",
                inputs=[dummy_component, dummy_component, dummy_component],
                outputs=[width, height],
                show_progress=False,
            )

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressImg2img",
                inputs=[dummy_component],
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            toprow.button_interrogate.click(
                fn=lambda *args: process_interrogate(interrogate, *args),
                **interrogate_args,
            )

            toprow.button_deepbooru.click(
                fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
                **interrogate_args,
            )

            toprow.token_button.click(fn=update_token_counter, inputs=[toprow.prompt, steps], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.negative_prompt, steps], outputs=[toprow.negative_token_counter])

            img2img_paste_fields = [
                (toprow.prompt, "Prompt"),
                (toprow.negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_name, "Sampler"),
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                *scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="img2img", source_text_component=toprow.prompt, source_image_component=None,
            ))
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        extra_networks_ui_img2img = ui_extra_networks.create_ui(img2img_interface, [img2img_generation_tab], 'img2img')
        ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)

        extra_tabs.__exit__()

    scripts.scripts_current = None

    if shared.backend == shared.Backend.DIFFUSERS:
        with gr.Blocks(analytics_enabled=False) as control_interface:
            from modules import ui_control
            ui_control.create_ui()
            timer.startup.record("ui-control")
    else:
        control_interface = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        from modules import ui_postprocessing
        ui_postprocessing.create_ui()
<<<<<<< HEAD
        timer.startup.record("ui-extras")

    with gr.Blocks(analytics_enabled=False) as train_interface:
        from modules import ui_train
        ui_train.create_ui([txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, cfg_scale, seed, width, height])
        timer.startup.record("ui-train")

    with gr.Blocks(analytics_enabled=False) as models_interface:
        from modules import ui_models
        ui_models.create_ui()
        timer.startup.record("ui-models")

    with gr.Blocks(analytics_enabled=False) as interrogate_interface:
        from modules import ui_interrogate
        ui_interrogate.create_ui()
        timer.startup.record("ui-interrogate")


    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)
        args = (info.component_args() if callable(info.component_args) else info.component_args) or {}
        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise ValueError(f'bad options item type: {t} for key {key}')
        elem_id = f"setting_{key}"

        if not is_quicksettings:
            dirtyable_setting = gr.Group(elem_classes="dirtyable", visible=args.get("visible", True))
            dirtyable_setting.__enter__()
            dirty_indicator = gr.Button("", elem_classes="modification-indicator", elem_id="modification_indicator_" + key)

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
            else:
                with FormRow():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                    ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        elif info.folder is not None:
            with FormRow():
                res = comp(label=info.label, value=fun(), elem_id=elem_id, elem_classes="folder-selector", **args)
                # ui_common.create_browse_button(res, f"folder_{key}")
        else:
            try:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
            except Exception as e:
                log.error(f'Error creating setting: {key} {e}')
                res = None

        if res is not None and not is_quicksettings:
            res.change(fn=None, inputs=res, _js=f'(val) => markIfModified("{key}", val)')
            dirty_indicator.click(fn=lambda: getattr(opts, key), outputs=res, show_progress=False)
            dirtyable_setting.__exit__()

        return res

    def create_dirty_indicator(key, keys_to_reset, **kwargs):
        def get_opt_values():
            return [getattr(opts, _key) for _key in keys_to_reset]

        elements_to_reset = [component_dict[_key] for _key in keys_to_reset if component_dict[_key] is not None]
        indicator = gr.Button("", elem_classes="modification-indicator", elem_id=f"modification_indicator_{key}", **kwargs)
        indicator.click(fn=get_opt_values, outputs=elements_to_reset, show_progress=False)
        return indicator

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config)
    components = []
    component_dict = {}
    shared.settings_components = component_dict
    dummy_component1 = gr.Label(visible=False)

    script_callbacks.ui_settings_callback()
    opts.reorder()

    def run_settings(*args):
        changed = []
        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component:
                continue
            if not opts.same_type(value, opts.data_labels[key].default):
                log.error(f'Setting bad value: {key}={value} expecting={type(opts.data_labels[key].default).__name__}')
                continue
            if opts.set(key, value):
                changed.append(key)
        if cmd_opts.use_directml:
            directml_override_opts()
        if cmd_opts.use_openvino:
            if not shared.opts.cuda_compile:
                shared.log.warning("OpenVINO: Enabling Torch Compile")
                shared.opts.cuda_compile = True
            if shared.opts.cuda_compile_backend != "openvino_fx":
                shared.log.warning("OpenVINO: Setting Torch Compiler backend to OpenVINO FX")
                shared.opts.cuda_compile_backend = "openvino_fx"
            if shared.opts.sd_backend != "diffusers":
                shared.log.warning("OpenVINO: Setting backend to Diffusers")
                shared.opts.sd_backend = "diffusers"
        try:
            opts.save(shared.config_filename)
            if len(changed) > 0:
                log.info(f'Settings: changed={len(changed)} {changed}')
        except RuntimeError:
            log.error(f'Settings failed: change={len(changed)} {changed}')
            return opts.dumpjson(), f'{len(changed)} Settings changed without save: {", ".join(changed)}'
        return opts.dumpjson(), f'{len(changed)} Settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()
        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()
        if cmd_opts.use_directml:
            directml_override_opts()
        opts.save(shared.config_filename)
        log.debug(f'Setting changed: key={key}, value={value}')
        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row(elem_id="system_row"):
            restart_submit = gr.Button(value="Restart server", variant='primary', elem_id="restart_submit")
            shutdown_submit = gr.Button(value="Shutdown server", variant='primary', elem_id="shutdown_submit")
            unload_sd_model = gr.Button(value='Unload checkpoint', variant='primary', elem_id="sett_unload_sd_model")
            reload_sd_model = gr.Button(value='Reload checkpoint', variant='primary', elem_id="sett_reload_sd_model")

        with gr.Tabs(elem_id="system") as system_tabs:
            global ui_system_tabs # pylint: disable=global-statement
            ui_system_tabs = system_tabs
            with gr.TabItem("Settings", id="system_settings", elem_id="tab_settings"):
                with gr.Row(elem_id="settings_row"):
                    settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
                    preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
                    defaults_submit = gr.Button(value="Restore defaults", variant='primary', elem_id="defaults_submit")
                with gr.Row():
                    _settings_search = gr.Text(label="Search", elem_id="settings_search")

                result = gr.HTML(elem_id="settings_result")
                quicksettings_names = opts.quicksettings_list
                quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}
                quicksettings_list = []

                previous_section = []
                tab_item_keys = []
                current_tab = None
                current_row = None
                with gr.Tabs(elem_id="settings"):
                    for i, (k, item) in enumerate(opts.data_labels.items()):
                        section_must_be_skipped = item.section[0] is None
                        if previous_section != item.section and not section_must_be_skipped:
                            elem_id, text = item.section
                            if current_tab is not None and len(previous_section) > 0:
                                create_dirty_indicator(previous_section[0], tab_item_keys)
                                tab_item_keys = []
                                current_row.__exit__()
                                current_tab.__exit__()
                            current_tab = gr.TabItem(elem_id=f"settings_{elem_id}", label=text)
                            current_tab.__enter__()
                            current_row = gr.Column(variant='compact')
                            current_row.__enter__()
                            previous_section = item.section
                        if k in quicksettings_names and not shared.cmd_opts.freeze:
                            quicksettings_list.append((i, k, item))
                            components.append(dummy_component)
                        elif section_must_be_skipped:
                            components.append(dummy_component)
                        else:
                            component = create_setting_component(k)
                            component_dict[k] = component
                            tab_item_keys.append(k)
                            components.append(component)
                    if current_tab is not None and len(previous_section) > 0:
                        create_dirty_indicator(previous_section[0], tab_item_keys)
                        tab_item_keys = []
                        current_row.__exit__()
                        current_tab.__exit__()

                    request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications", visible=False)
                    with gr.TabItem("Show all pages", elem_id="settings_show_all_pages"):
                        create_dirty_indicator("show_all_pages", [], interactive=False)

            with gr.TabItem("User interface", id="system_config", elem_id="tab_config"):
                loadsave.create_ui()
                create_dirty_indicator("tab_defaults", [], interactive=False)

            with gr.TabItem("Change log", id="change_log", elem_id="system_tab_changelog"):
                with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
                    md = f.read()
                gr.Markdown(md)

            with gr.TabItem("Licenses", id="system_licenses", elem_id="system_tab_licenses"):
                gr.HTML(shared.html("licenses.html"), elem_id="licenses", elem_classes="licenses")
                create_dirty_indicator("tab_licenses", [], interactive=False)

        def unload_sd_weights():
            modules.sd_models.unload_model_weights(op='model')
            modules.sd_models.unload_model_weights(op='refiner')

        def reload_sd_weights():
            modules.sd_models.reload_model_weights()

        unload_sd_model.click(fn=unload_sd_weights, inputs=[], outputs=[])
        reload_sd_model.click(fn=reload_sd_weights, inputs=[], outputs=[])
        request_notifications.click(fn=lambda: None, inputs=[], outputs=[], _js='function(){}')
        preview_theme.click(fn=None, _js='previewTheme', inputs=[], outputs=[])

    timer.startup.record("ui-settings")
=======

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=image,
                    ))

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    modelmerger_ui = ui_checkpoint_merger.UiCheckpointMerger()

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

        with gr.Row(variant="compact", equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):

                with gr.Tab(label="Create embedding", id="create_embedding"):
                    new_embedding_name = gr.Textbox(label="Name", elem_id="train_new_embedding_name")
                    initialization_text = gr.Textbox(label="Initialization text", value="*", elem_id="train_initialization_text")
                    nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1, elem_id="train_nvpt")
                    overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding", elem_id="train_overwrite_old_embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="Create embedding", variant='primary', elem_id="train_create_embedding")

                with gr.Tab(label="Create hypernetwork", id="create_hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(label="Name", elem_id="train_new_hypernetwork_name")
                    new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], elem_id="train_new_hypernetwork_sizes")
                    new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'", elem_id="train_new_hypernetwork_layer_structure")
                    new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)", choices=hypernetworks_ui.keys, elem_id="train_new_hypernetwork_activation_func")
                    new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"], elem_id="train_new_hypernetwork_initialization_option")
                    new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization", elem_id="train_new_hypernetwork_add_layer_norm")
                    new_hypernetwork_use_dropout = gr.Checkbox(label="Use dropout", elem_id="train_new_hypernetwork_use_dropout")
                    new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0", label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15", placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
                    overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork", elem_id="train_overwrite_old_hypernetwork")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary', elem_id="train_create_hypernetwork")

                def get_textual_inversion_template_names():
                    return sorted(textual_inversion.textual_inversion_templates)

                with gr.Tab(label="Train", id="train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
                    with FormRow():
                        train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")

                        train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork", choices=sorted(shared.hypernetworks))
                        create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks, lambda: {"choices": sorted(shared.hypernetworks)}, "refresh_train_hypernetwork_name")

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005", elem_id="train_embedding_learn_rate")
                        hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001", elem_id="train_hypernetwork_learn_rate")

                    with FormRow():
                        clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                        clip_grad_value = gr.Textbox(placeholder="Gradient clip value", value="0.1", show_label=False)

                    with FormRow():
                        batch_size = gr.Number(label='Batch size', value=1, precision=0, elem_id="train_batch_size")
                        gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0, elem_id="train_gradient_step")

                    dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images", elem_id="train_dataset_directory")
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion", elem_id="train_log_directory")

                    with FormRow():
                        template_file = gr.Dropdown(label='Prompt template', value="style_filewords.txt", elem_id="train_template_file", choices=get_textual_inversion_template_names())
                        create_refresh_button(template_file, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")

                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_training_height")
                    varsize = gr.Checkbox(label="Do not resize images", value=False, elem_id="train_varsize")
                    steps = gr.Number(label='Max steps', value=100000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_save_embedding_every")

                    use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False, elem_id="use_weight")

                    save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="Train Embedding", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="Interrupt", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id='ti_gallery_container'):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                gr.Gallery(label='Output', show_label=False, elem_id='ti_gallery', columns=4)
                gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=textual_inversion_ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ]
        )

        create_hypernetwork.click(
            fn=hypernetworks_ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(textual_inversion_ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(hypernetworks_ui.train_hypernetwork, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)

    settings = ui_settings.UiSettings()
    settings.create_ui(loadsave, dummy_component)

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "train"),
    ]
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    interfaces = []
    interfaces += [(txt2img_interface, "Text", "txt2img")]
    interfaces += [(img2img_interface, "Image", "img2img")]
    interfaces += [(control_interface, "Control", "control")] if control_interface is not None else []
    interfaces += [(extras_interface, "Process", "process")]
    interfaces += [(interrogate_interface, "Interrogate", "interrogate")]
    interfaces += [(train_interface, "Train", "train")]
    interfaces += [(models_interface, "Models", "models")]
    interfaces += script_callbacks.ui_tabs_callback()
<<<<<<< HEAD
    interfaces += [(settings_interface, "System", "system")]
=======
    interfaces += [(settings.interface, "Settings", "settings")]
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    from modules import ui_extensions
    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]
    timer.startup.record("ui-extensions")

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

<<<<<<< HEAD
    with gr.Blocks(theme=theme.gradio_theme, analytics_enabled=False, title="SD.Next") as demo:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for _i, k, _item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component
=======
    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        settings.add_quicksettings()
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        generation_parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
<<<<<<< HEAD
            for interface, label, ifid in interfaces:
                if interface is None:
                    continue
                # if label in shared.opts.hidden_tabs or label == '':
                #    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    # log.debug(f'UI render: id={ifid}')
                    interface.render()
            for interface, _label, ifid in interfaces:
                if interface is None:
                    continue
                if ifid in ["extensions", "system"]:
                    continue
                loadsave.add_block(interface, ifid)
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()
        if opts.notification_audio_enable and os.path.exists(os.path.join(script_path, opts.notification_audio_path)):
            gr.Audio(interactive=False, value=os.path.join(script_path, opts.notification_audio_path), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        components = [c for c in components if c is not None]
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )
        defaults_submit.click(fn=lambda: shared.restore_defaults(restart=True), _js="restartReload")
        restart_submit.click(fn=lambda: shared.restart_server(restart=True), _js="restartReload")
        shutdown_submit.click(fn=lambda: shared.restart_server(restart=False), _js="restartReload")

        for _i, k, _item in quicksettings_list:
            component = component_dict[k]
            info = opts.data_labels[k]
            change_handler = component.release if hasattr(component, 'release') else component.change
            change_handler(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
                show_progress=info.refresh is not None,
            )

        button_set_checkpoint = gr.Button('Change model', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_checkpoint'], dummy_component],
            outputs=[component_dict['sd_model_checkpoint'], text_settings],
        )
        button_set_refiner = gr.Button('Change refiner', elem_id='change_refiner', visible=False)
        button_set_refiner.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_refiner'], dummy_component],
            outputs=[component_dict['sd_model_refiner'], text_settings],
        )
        button_set_vae = gr.Button('Change VAE', elem_id='change_vae', visible=False)
        button_set_vae.click(
            fn=lambda value, _: run_settings_single(value, key='sd_vae'),
            _js="function(v){ var res = desiredVAEName; desiredVAEName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_vae'], dummy_component],
            outputs=[component_dict['sd_vae'], text_settings],
        )

        def reference_submit(model):
            if '@' not in model: # diffusers
                loaded = modelloader.load_reference(model)
                return model if loaded else opts.sd_model_checkpoint
            else: # civitai
                model, url = model.split('@')
                loaded = modelloader.load_civitai(model, url)
                return loaded if loaded is not None else opts.sd_model_checkpoint

        button_set_reference = gr.Button('Change reference', elem_id='change_reference', visible=False)
        button_set_reference.click(
            fn=reference_submit,
            _js="function(v){ return desiredCheckpointName; }",
            inputs=[component_dict['sd_model_checkpoint']],
            outputs=[component_dict['sd_model_checkpoint']],
        )
        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys if component_dict[k] is not None],
            queue=False,
        )

    timer.startup.record("ui-defaults")
    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave
    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'


def html_head():
    head = ''
    main = ['script.js']
    for js in main:
        script_js = os.path.join(script_path, "javascript", js)
        head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'
    added = []
    for script in modules.scripts.list_scripts("javascript", ".js"):
        if script.filename in main:
            continue
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # log.debug(f'Adding JS scripts: {added}')
    return head


def html_body():
    body = ''
    inline = ''
    if opts.theme_style != 'Auto':
        inline += f"set_theme('{opts.theme_style.lower()}');"
    body += f'<script type="text/javascript">{inline}</script>\n'
    return body


def html_css(is_builtin: bool):
    added = []

    def stylesheet(fn):
        added.append(fn)
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    css = 'sdnext.css' if is_builtin else 'base.css'
    head = stylesheet(os.path.join(script_path, 'javascript', css))
    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)
    if opts.gradio_theme in theme.list_builtin_themes():
        head += stylesheet(os.path.join(script_path, "javascript", f"{opts.gradio_theme}.css"))
    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # log.debug(f'Adding CSS stylesheets: {added}')
    return head


def reload_javascript():
    is_builtin = theme.reload_gradio_theme()
    head = html_head()
    css = html_css(is_builtin)
    body = html_body()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}{body}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def setup_ui_api(app):
    from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
    from typing import List

    class QuicksettingsHint(BaseModel): # pylint: disable=too-few-public-methods
=======
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            loadsave.setup_ui()

        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")

        settings.add_functionality(demo)

        update_image_cfg_scale_visibility = lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
        settings.text_settings.change(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])
        demo.load(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[image_cfg_scale])

        modelmerger_ui.setup_ui(dummy_component=dummy_component, sd_model_checkpoint_component=settings.component_dict['sd_model_checkpoint'])

    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

<<<<<<< HEAD
    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=List[QuicksettingsHint])
    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
=======
    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
