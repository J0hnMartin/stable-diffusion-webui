from __future__ import annotations
import base64
import io
<<<<<<< HEAD
import os
import re
import json
from PIL import Image
import gradio as gr
from modules.paths import data_path
from modules import shared, ui_tempdir, script_callbacks, images

=======
import json
import os
import re

import gradio as gr
from modules.paths import data_path
from modules import shared, ui_tempdir, script_callbacks, processing
from PIL import Image
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$") # pylint: disable=anomalous-backslash-in-string
type_of_gr_update = type(gr.update())
<<<<<<< HEAD
paste_fields = {}
registered_param_bindings = []
debug = shared.log.trace if os.environ.get('SD_PASTE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PASTE')
=======
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


class ParamBinding:
    def __init__(self, paste_button, tabname, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []
<<<<<<< HEAD
        debug(f'ParamBinding: {vars(self)}')
=======


paste_fields: dict[str, dict] = {}
registered_param_bindings: list[ParamBinding] = []
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def reset():
    paste_fields.clear()
    registered_param_bindings.clear()


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text
    return json.dumps(text, ensure_ascii=False)

<<<<<<< HEAD
=======
    return json.dumps(text, ensure_ascii=False)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
<<<<<<< HEAD
=======

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    try:
        return json.loads(text)
    except Exception:
        return text


def image_from_url_text(filedata):
    if filedata is None:
        return None
<<<<<<< HEAD
    if type(filedata) == list and len(filedata) > 0 and type(filedata[0]) == dict and filedata[0].get("is_file", False):
=======

    if type(filedata) == list and filedata and type(filedata[0]) == dict and filedata[0].get("is_file", False):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        filedata = filedata[0]
    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]
        is_in_right_dir = ui_tempdir.check_tmp_file(shared.demo, filename)
<<<<<<< HEAD
        if is_in_right_dir:
            filename = filename.rsplit('?', 1)[0]
            if not os.path.exists(filename):
                shared.log.error(f'Image file not found: {filename}')
                image = Image.new('RGB', (512, 512))
                image.info['parameters'] = f'Image file not found: {filename}'
                return image
            image = Image.open(filename)
            geninfo, _items = images.read_info_from_image(image)
            image.info['parameters'] = geninfo
            return image
        else:
            shared.log.warning(f'File access denied: {filename}')
            return None
=======
        assert is_in_right_dir, 'trying to open image file outside of allowed directories'

        filename = filename.rsplit('?', 1)[0]
        return Image.open(filename)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    if type(filedata) == list:
        if len(filedata) == 0:
            return None
        filedata = filedata[0]
    if type(filedata) == dict:
        shared.log.warning('Incorrect filedata received')
        return None
    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]
    if filedata.startswith("data:image/webp;base64,"):
        filedata = filedata[len("data:image/webp;base64,"):]
    if filedata.startswith("data:image/jpeg;base64,"):
        filedata = filedata[len("data:image/jpeg;base64,"):]
    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    images.read_info_from_image(image)
    return image


def add_paste_fields(tabname, init_img, fields, override_settings_component=None):
    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}
    # backwards compatibility for existing extensions
    import modules.ui
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields
    elif tabname == 'img2img':
        modules.ui.img2img_paste_fields = fields


def create_buttons(tabs_list):
    buttons = {}
    for tab in tabs_list:
        name = tab
        if name == 'txt2img':
            name = 'Text'
        elif name == 'img2img':
            name = 'Image'
        elif name == 'inpaint':
            name = 'Inpaint'
        elif name == 'extras':
            name = 'Process'
        elif name == 'control':
            name = 'Control'
        buttons[tab] = gr.Button(f"➠ {name}", elem_id=f"{tab}_tab")
    return buttons


def bind_buttons(buttons, send_image, send_generate_info):
    """old function for backwards compatibility; do not use this, use register_paste_params_button"""
    for tabname, button in buttons.items():
        source_text_component = send_generate_info if isinstance(send_generate_info, gr.components.Component) else None
        source_tabname = send_generate_info if isinstance(send_generate_info, str) else None
        bindings = ParamBinding(paste_button=button, tabname=tabname, source_text_component=source_text_component, source_image_component=send_image, source_tabname=source_tabname)
        register_paste_params_button(bindings)


def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


def connect_paste_params_buttons():
    for binding in registered_param_bindings:
        if binding.tabname not in paste_fields:
            debug(f"Not not registered: tab={binding.tabname}")
            continue
        destination_image_component = paste_fields[binding.tabname]["init_img"]
        fields = paste_fields[binding.tabname]["fields"]
        override_settings_component = binding.override_settings_component or paste_fields[binding.tabname]["override_settings_component"]
        destination_width_component = next(iter([field for field, name in fields if name == "Size-1"] if fields else []), None)
        destination_height_component = next(iter([field for field, name in fields if name == "Size-2"] if fields else []), None)

        if binding.source_image_component and destination_image_component:
            if isinstance(binding.source_image_component, gr.Gallery):
                func = send_image_and_dimensions if destination_width_component else image_from_url_text
                jsfunc = "extract_image_from_gallery"
            else:
                func = send_image_and_dimensions if destination_width_component else lambda x: x
                jsfunc = None
            binding.paste_button.click(
                fn=func,
                _js=jsfunc,
                inputs=[binding.source_image_component],
                outputs=[destination_image_component, destination_width_component, destination_height_component] if destination_width_component else [destination_image_component],
                show_progress=False,
            )
        if binding.source_text_component is not None and fields is not None:
            connect_paste(binding.paste_button, fields, binding.source_text_component, override_settings_component, binding.tabname)
        if binding.source_tabname is not None and fields is not None:
            paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration'] + (["Seed"] if shared.opts.send_seed else []) + binding.paste_field_names
            binding.paste_button.click(
                fn=lambda *x: x,
                inputs=[field for field, name in paste_fields[binding.source_tabname]["fields"] if name in paste_field_names],
                outputs=[field for field, name in fields if name in paste_field_names],
                show_progress=False,
            )
        binding.paste_button.click(
            fn=None,
            _js=f"switch_to_{binding.tabname}",
<<<<<<< HEAD
            inputs=[],
            outputs=[],
=======
            inputs=None,
            outputs=None,
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            show_progress=False,
        )


def send_image_and_dimensions(x):
    img = x if isinstance(x, Image.Image) else image_from_url_text(x)
    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()
    return img, w, h


<<<<<<< HEAD
def find_hypernetwork_key(hypernet_name, hypernet_hash=None):
    """Determines the config parameter name to use for the hypernet based on the parameters in the infotext.
    Example: an infotext provides "Hypernet: ke-ta" and "Hypernet hash: 1234abcd". For the "Hypernet" config
    parameter this means there should be an entry that looks like "ke-ta-10000(1234abcd)" to set it to.
    If the infotext has no hash, then a hypernet with the same name will be selected instead.
    """
    hypernet_name = hypernet_name.lower()
    if hypernet_hash is not None:
        # Try to match the hash in the name
        for hypernet_key in shared.hypernetworks.keys():
            result = re_hypernet_hash.search(hypernet_key)
            if result is not None and result[1] == hypernet_hash:
                return hypernet_key
    else:
        # Fall back to a hypernet with the same name
        for hypernet_key in shared.hypernetworks.keys():
            if hypernet_key.lower().startswith(hypernet_name):
                return hypernet_key

    return None
=======
def restore_old_hires_fix_params(res):
    """for infotexts that specify old First pass size parameter, convert it into
    width, height, and hr scale"""

    firstpass_width = res.get('First pass size-1', None)
    firstpass_height = res.get('First pass size-2', None)

    if shared.opts.use_old_hires_fix_width_height:
        hires_width = int(res.get("Hires resize-1", 0))
        hires_height = int(res.get("Hires resize-2", 0))

        if hires_width and hires_height:
            res['Size-1'] = hires_width
            res['Size-2'] = hires_height
            return

    if firstpass_width is None or firstpass_height is None:
        return

    firstpass_width, firstpass_height = int(firstpass_width), int(firstpass_height)
    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    if firstpass_width == 0 or firstpass_height == 0:
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    res['Size-1'] = firstpass_width
    res['Size-2'] = firstpass_height
    res['Hires resize-1'] = width
    res['Hires resize-2'] = height
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def parse_generation_parameters(x: str):
    res = {}
<<<<<<< HEAD
    if x is None:
        return res
    remaining = x.replace('\n', ' ').strip()
    if len(remaining) == 0:
        return res
    remaining = x[7:] if x.startswith('Prompt: ') else x
    remaining = x[11:] if x.startswith('parameters: ') else x
    if 'Steps: ' in remaining and 'Negative prompt: ' not in remaining:
        remaining = remaining.replace('Steps: ', 'Negative prompt: Steps: ')
    prompt, remaining = remaining.strip().split('Negative prompt: ', maxsplit=1) if 'Negative prompt: ' in remaining else (remaining, '')
    res["Prompt"] = prompt.strip()
    negative, remaining = remaining.strip().split('Steps: ', maxsplit=1) if 'Steps: ' in remaining else (remaining, None)
    res["Negative prompt"] = negative.strip()
    if remaining is None:
        return res
    remaining = f'Steps: {remaining}'
    for k, v in re_param.findall(remaining.strip()):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)
=======

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if shared.opts.infotext_styles != "Ignore":
        found_styles, prompt, negative_prompt = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

        if shared.opts.infotext_styles == "Apply":
            res["Styles array"] = found_styles
        elif shared.opts.infotext_styles == "Apply if any" and found_styles:
            res["Styles array"] = found_styles

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            m = re_imagesize.match(v)
            if m is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
<<<<<<< HEAD
            pass
    if res.get('VAE', None) == 'TAESD':
        res["Full quality"] = False
    debug(f"Parse prompt: {res}")
=======
            print(f"Error parsing \"{k}: {v}\"")

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    skip = set(shared.opts.infotext_skip_pasting)
    res = {k: v for k, v in res.items() if k not in skip}

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    return res


infotext_to_setting_name_mapping = [

<<<<<<< HEAD
infotext_to_setting_name_mapping = [
    ('Backend', 'sd_backend'),
    ('Model hash', 'sd_model_checkpoint'),
    ('Refiner', 'sd_model_refiner'),
    ('VAE', 'sd_vae'),
    ('Parser', 'prompt_attention'),
    ('Color correction', 'img2img_color_correction'),
    # Samplers
    ('Sampler Eta', 'scheduler_eta'),
    ('Sampler ENSD', 'eta_noise_seed_delta'),
    ('Sampler order', 'schedulers_solver_order'),
    # Samplers diffusers
    ('Sampler beta schedule', 'schedulers_beta_schedule'),
    ('Sampler beta start', 'schedulers_beta_start'),
    ('Sampler beta end', 'schedulers_beta_end'),
    ('Sampler DPM solver', 'schedulers_dpm_solver'),
    # Samplers original
    ('Sampler brownian', 'schedulers_brownian_noise'),
    ('Sampler discard', 'schedulers_discard_penultimate'),
    ('Sampler dyn threshold', 'schedulers_use_thresholding'),
    ('Sampler karras', 'schedulers_use_karras'),
    ('Sampler low order', 'schedulers_use_loworder'),
    ('Sampler quantization', 'enable_quantization'),
    ('Sampler sigma', 'schedulers_sigma'),
    ('Sampler sigma min', 's_min'),
    ('Sampler sigma max', 's_max'),
    ('Sampler sigma churn', 's_churn'),
    ('Sampler sigma uncond', 's_min_uncond'),
    ('Sampler sigma noise', 's_noise'),
    ('Sampler sigma tmin', 's_tmin'),
    ('Sampler ENSM', 'initial_noise_multiplier'), # img2img only
    ('UniPC skip type', 'uni_pc_skip_type'),
    ('UniPC variant', 'uni_pc_variant'),
    # Token Merging
    ('Mask weight', 'inpainting_mask_weight'),
    ('Token merging ratio', 'token_merging_ratio'),
    ('ToMe', 'token_merging_ratio'),
    ('ToMe hires', 'token_merging_ratio_hr'),
    ('ToMe img2img', 'token_merging_ratio_img2img'),
=======
]
"""Mapping of infotext labels to setting names. Only left for backwards compatibility - use OptionInfo(..., infotext='...') instead.
Example content:

infotext_to_setting_name_mapping = [
    ('Conditional mask weight', 'inpainting_mask_weight'),
    ('Model hash', 'sd_model_checkpoint'),
    ('ENSD', 'eta_noise_seed_delta'),
    ('Schedule type', 'k_sched_type'),
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
]
"""


def create_override_settings_dict(text_pairs):
    res = {}
    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)
        params[k] = v.strip()
<<<<<<< HEAD
    for param_name, setting_name in infotext_to_setting_name_mapping:
=======

    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
    for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        value = params.get(param_name, None)
        if value is None:
            continue
        res[setting_name] = shared.opts.cast_value(setting_name, value)
    return res


def connect_paste(button, local_paste_fields, input_comp, override_settings_component, tabname):

    def paste_func(prompt):
        if prompt is None or len(prompt.strip()) == 0 and not shared.cmd_opts.hide_ui_dir_config:
            filename = os.path.join(data_path, "params.txt")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf8") as file:
                    prompt = file.read()
                shared.log.debug(f'Paste prompt: type="params" prompt="{prompt}"')
            else:
                prompt = ''
        else:
            shared.log.debug(f'Paste prompt: type="current" prompt="{prompt}"')
        params = parse_generation_parameters(prompt)
        script_callbacks.infotext_pasted_callback(prompt, params)
        res = []
        applied = {}
        for output, key in local_paste_fields:
            if callable(key):
                v = key(params)
            else:
                v = params.get(key, None)
            if v is None:
                res.append(gr.update())
            elif isinstance(v, type_of_gr_update):
                res.append(v)
                applied[key] = v
            else:
                try:
                    valtype = type(output.value)
                    if valtype == bool and v == "False":
                        val = False
                    else:
                        val = valtype(v)
                    res.append(gr.update(value=val))
                    applied[key] = val
                except Exception:
                    res.append(gr.update())
        debug(f"Parse apply: {applied}")
        return res

    if override_settings_component is not None:
        already_handled_fields = {key: 1 for _, key in paste_fields}

        def paste_settings(params):
            vals = {}
<<<<<<< HEAD
            for param_name, setting_name in infotext_to_setting_name_mapping:
=======

            mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
            for param_name, setting_name in mapping + infotext_to_setting_name_mapping:
                if param_name in already_handled_fields:
                    continue

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                v = params.get(param_name, None)
                if v is None:
                    continue
                if shared.opts.disable_weights_auto_swap:
                    if setting_name == "sd_model_checkpoint" or setting_name == 'sd_model_refiner' or setting_name == 'sd_backend' or setting_name == 'sd_vae':
                        continue
                v = shared.opts.cast_value(setting_name, v)
                current_value = getattr(shared.opts, setting_name, None)
                if v == current_value:
                    continue
                if type(current_value) == str and v == os.path.splitext(current_value)[0]:
                    continue
                vals[param_name] = v
            vals_pairs = [f"{k}: {v}" for k, v in vals.items()]
<<<<<<< HEAD
            shared.log.debug(f'Settings overrides: {vals_pairs}')
            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=len(vals_pairs) > 0)
        local_paste_fields = local_paste_fields + [(override_settings_component, paste_settings)]
=======

            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=bool(vals_pairs))

        paste_fields = paste_fields + [(override_settings_component, paste_settings)]
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    button.click(
        fn=paste_func,
        inputs=[input_comp],
<<<<<<< HEAD
        outputs=[x[0] for x in local_paste_fields],
=======
        outputs=[x[0] for x in paste_fields],
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        show_progress=False,
    )
    button.click(
        fn=None,
        _js=f"recalculate_prompts_{tabname}",
        inputs=[],
        outputs=[],
        show_progress=False,
    )
<<<<<<< HEAD
=======

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
