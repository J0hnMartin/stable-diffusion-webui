import json
import html
import os
import shutil
import platform
import subprocess
import gradio as gr
from modules import call_queue, shared
from modules.generation_parameters_copypaste import image_from_url_text, parse_generation_parameters
import modules.ui_symbols as symbols
import modules.images
<<<<<<< HEAD
import modules.script_callbacks


folder_symbol = symbols.folder
debug = shared.log.trace if os.environ.get('SD_PASTE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PASTE')
=======
from modules.ui_components import ToolButton
import modules.generation_parameters_copypaste as parameters_copypaste

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, generation_info
        infotext = generation_info["infotexts"][img_index]
        html_info_formatted = infotext_to_html(infotext)
        return html_info, html_info_formatted
    except Exception:
        pass
    return html_info, html_info


<<<<<<< HEAD
def plaintext_to_html(text):
    res = '<p class="plaintext">' + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + '</p>'
    return res
=======
def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def infotext_to_html(text):
    res = parse_generation_parameters(text)
    prompt = res.get('Prompt', '')
    negative = res.get('Negative prompt', '')
    res.pop('Prompt', None)
    res.pop('Negative prompt', None)
    params = [f'{k}: {v}' for k, v in res.items() if v is not None]
    params = '| '.join(params) if len(params) > 0 else ''
    code = f'''
        <p><b>Prompt:</b> {html.escape(prompt)}</p>
        <p><b>Negative:</b> {html.escape(negative)}</p>
        <p><b>Parameters:</b> {html.escape(params)}</p>
        '''
    return code


def delete_files(js_data, images, _html_info, index):
    try:
        data = json.loads(js_data)
    except Exception:
        data = { 'index_of_first_image': 0 }
    start_index = 0
<<<<<<< HEAD
    if index > -1 and shared.opts.save_selected_only and (index >= data['index_of_first_image']):
=======
    only_one = False

    if index > -1 and shared.opts.save_selected_only and (index >= data["index_of_first_image"]):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only
        only_one = True
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        images = [images[index]]
        start_index = index
        filenames = []
    filenames = []
    fullfns = []
    for _image_index, filedata in enumerate(images, start_index):
        if 'name' in filedata and os.path.isfile(filedata['name']):
            fullfn = filedata['name']
            filenames.append(os.path.basename(fullfn))
            try:
                os.remove(fullfn)
                base, _ext = os.path.splitext(fullfn)
                desc = f'{base}.txt'
                if os.path.exists(desc):
                    os.remove(desc)
                fullfns.append(fullfn)
                shared.log.info(f"Deleting image: {fullfn}")
            except Exception as e:
                shared.log.error(f'Error deleting file: {fullfn} {e}')
    images = [image for image in images if image['name'] not in fullfns]
    return images, plaintext_to_html(f"Deleted: {filenames[0] if len(filenames) > 0 else 'none'}")


def save_files(js_data, images, html_info, index):
    os.makedirs(shared.opts.outdir_save, exist_ok=True)

    class PObject: # pylint: disable=too-few-public-methods
        def __init__(self, d=None):
            if d is not None:
                for k, v in d.items():
                    setattr(self, k, v)
            self.prompt = getattr(self, 'prompt', None) or getattr(self, 'Prompt', None)
            self.all_prompts = getattr(self, 'all_prompts', [self.prompt])
            self.negative_prompt = getattr(self, 'negative_prompt', None)
            self.all_negative_prompt = getattr(self, 'all_negative_prompts', [self.negative_prompt])
            self.seed = getattr(self, 'seed', None) or getattr(self, 'Seed', None)
            self.all_seeds = getattr(self, 'all_seeds', [self.seed])
            self.subseed = getattr(self, 'subseed', None)
            self.all_subseeds = getattr(self, 'all_subseeds', [self.subseed])
            self.width = getattr(self, 'width', None)
            self.height = getattr(self, 'height', None)
            self.index_of_first_image = getattr(self, 'index_of_first_image', 0)
            self.infotexts = getattr(self, 'infotexts', [html_info])
            self.infotext = self.infotexts[0] if len(self.infotexts) > 0 else html_info
            self.outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
    try:
        data = json.loads(js_data)
    except Exception:
        data = {}
    p = PObject(data)
    start_index = 0
    if index > -1 and shared.opts.save_selected_only and (index >= p.index_of_first_image):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only # pylint: disable=no-member
        images = [images[index]]
        start_index = index
    filenames = []
    fullfns = []
    for image_index, filedata in enumerate(images, start_index):
        is_grid = image_index < p.index_of_first_image # pylint: disable=no-member
        i = 0 if is_grid else (image_index - p.index_of_first_image) # pylint: disable=no-member
        while len(p.all_seeds) <= i:
            p.all_seeds.append(p.seed)
        while len(p.all_prompts) <= i:
            p.all_prompts.append(p.prompt)
        while len(p.infotexts) <= i:
            p.infotexts.append(p.infotext)
        if 'name' in filedata and ('tmp' not in filedata['name']) and os.path.isfile(filedata['name']):
            fullfn = filedata['name']
            filenames.append(os.path.basename(fullfn))
            fullfns.append(fullfn)
            destination = shared.opts.outdir_save
            namegen = modules.images.FilenameGenerator(p, seed=p.all_seeds[i], prompt=p.all_prompts[i], image=None)  # pylint: disable=no-member
            dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
            destination = os.path.join(destination, dirname)
            destination = namegen.sanitize(destination)
            os.makedirs(destination, exist_ok = True)
            shutil.copy(fullfn, destination)
            shared.log.info(f'Copying image: file="{fullfn}" folder="{destination}"')
            tgt_filename = os.path.join(destination, os.path.basename(fullfn))
            if shared.opts.save_txt:
                try:
                    from PIL import Image
                    image = Image.open(fullfn)
                    info, _ = images.read_info_from_image(image)
                    filename_txt = f"{os.path.splitext(tgt_filename)[0]}.txt"
                    with open(filename_txt, "w", encoding="utf8") as file:
                        file.write(f"{info}\n")
                    shared.log.debug(f'Saving: text="{filename_txt}"')
                except Exception as e:
                    shared.log.warning(f'Image description save failed: {filename_txt} {e}')
            modules.script_callbacks.image_save_btn_callback(tgt_filename)
        else:
            image = image_from_url_text(filedata)
<<<<<<< HEAD
            info = p.infotexts[i + 1] if len(p.infotexts) > len(p.all_seeds) else p.infotexts[i] # infotexts may be offset by 1 because the first image is the grid
            fullfn, txt_fullfn = modules.images.save_image(image, shared.opts.outdir_save, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], info=info, extension=shared.opts.samples_format, grid=is_grid, p=p)
            if fullfn is None:
                continue
            filename = os.path.relpath(fullfn, shared.opts.outdir_save)
=======

            is_grid = image_index < p.index_of_first_image
            i = 0 if is_grid else (image_index - p.index_of_first_image)

            p.batch_index = image_index-1
            fullfn, txt_fullfn = modules.images.save_image(image, path, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], extension=extension, info=p.infotexts[image_index], grid=is_grid, p=p, save_to_dirs=save_to_dirs)

            filename = os.path.relpath(fullfn, path)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
<<<<<<< HEAD
                # fullfns.append(txt_fullfn)
            modules.script_callbacks.image_save_btn_callback(filename)
    if shared.opts.samples_save_zip and len(fullfns) > 1:
        zip_filepath = os.path.join(shared.opts.outdir_save, "images.zip")
=======
                fullfns.append(txt_fullfn)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler_name"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

    # Make Zip
    if do_make_zip:
        zip_fileseed = p.all_seeds[index-1] if only_one else p.all_seeds[0]
        namegen = modules.images.FilenameGenerator(p, zip_fileseed, p.all_prompts[0], image, True)
        zip_filename = namegen.apply(shared.opts.grid_zip_filename_pattern or "[datetime]_[[model_name]]_[seed]-[seed_last]")
        zip_filepath = os.path.join(path, f"{zip_filename}.zip")

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        from zipfile import ZipFile
        with ZipFile(zip_filepath, "w") as zip_file:
            for i in range(len(fullfns)):
                if os.path.isfile(fullfns[i]):
                    with open(fullfns[i], mode="rb") as f:
                        zip_file.writestr(filenames[i], f.read())
        fullfns.insert(0, zip_filepath)
    return gr.File.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0] if len(filenames) > 0 else 'none'}")


<<<<<<< HEAD
def open_folder(result_gallery, gallery_index = 0):
    try:
        folder = os.path.dirname(result_gallery[gallery_index]['name'])
    except Exception:
        folder = shared.opts.outdir_samples
    if not os.path.exists(folder):
        shared.log.warning(f'Folder open: folder={folder} does not exist')
        return
    elif not os.path.isdir(folder):
        shared.log.warning(f"Folder open: folder={folder} not a folder")
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(folder)
        if platform.system() == "Windows":
            os.startfile(path) # pylint: disable=no-member
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path]) # pylint: disable=consider-using-with
        elif "microsoft-standard-WSL2" in platform.uname().release:
            subprocess.Popen(["wsl-open", path]) # pylint: disable=consider-using-with
        else:
            subprocess.Popen(["xdg-open", path]) # pylint: disable=consider-using-with


def create_output_panel(tabname, preview=True):
    import modules.generation_parameters_copypaste as parameters_copypaste

    with gr.Column(variant='panel', elem_id=f"{tabname}_results"):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            if tabname == "txt2img":
                gr.HTML(value="", elem_id="main_info", visible=False, elem_classes=["main-info"])
            # columns are for <576px, <768px, <992px, <1200px, <1400px, >1400px
            result_gallery = gr.Gallery(value=[], label='Output', show_label=False, show_download_button=True, allow_preview=True, elem_id=f"{tabname}_gallery", container=False, preview=preview, columns=5, object_fit='scale-down', height=shared.opts.gallery_height or None)

        with gr.Column(elem_id=f"{tabname}_footer", elem_classes="gallery_footer"):
            dummy_component = gr.Label(visible=False)
            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                if not shared.cmd_opts.listen:
                    open_folder_button = gr.Button('Show', visible=not shared.cmd_opts.hide_ui_dir_config, elem_id=f'open_folder_{tabname}')
                    open_folder_button.click(open_folder, _js="(gallery, dummy) => [gallery, selected_gallery_index()]", inputs=[result_gallery, dummy_component], outputs=[])
                else:
                    clip_files = gr.Button('Copy', elem_id=f'open_folder_{tabname}')
                    clip_files.click(fn=None, _js='clip_gallery_urls', inputs=[result_gallery], outputs=[])
                save = gr.Button('Save', elem_id=f'save_{tabname}')
                delete = gr.Button('Delete', elem_id=f'delete_{tabname}')
                if shared.backend == shared.Backend.ORIGINAL:
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])
                else:
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "control", "extras"])

            download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')
            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext", visible=False) # contains raw infotext as returned by wrapped call
                html_info_formatted = gr.HTML(elem_id=f'html_info_formatted_{tabname}', elem_classes="infotext", visible=True) # contains html formatted infotext
                html_info.change(fn=infotext_to_html, inputs=[html_info], outputs=[html_info_formatted], show_progress=False)
=======
def create_output_panel(tabname, outdir, toprow=None):

    def open_folder(f):
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(elem_id=f"{tabname}_results"):
        if toprow:
            toprow.create_inline_toprow_image()

        with gr.Column(variant='panel', elem_id=f"{tabname}_results_panel"):
            with gr.Group(elem_id=f"{tabname}_gallery_container"):
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery", columns=4, preview=True, height=shared.opts.gallery_height or None)

            generation_info = None
            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                open_folder_button = ToolButton(folder_symbol, elem_id=f'{tabname}_open_folder', visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open images output directory.")

                if tabname != "extras":
                    save = ToolButton('💾', elem_id=f'save_{tabname}', tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).")
                    save_zip = ToolButton('🗃️', elem_id=f'save_zip_{tabname}', tooltip=f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})")

                buttons = {
                    'img2img': ToolButton('🖼️', elem_id=f'{tabname}_send_to_img2img', tooltip="Send image and generation parameters to img2img tab."),
                    'inpaint': ToolButton('🎨️', elem_id=f'{tabname}_send_to_inpaint', tooltip="Send image and generation parameters to img2img inpaint tab."),
                    'extras': ToolButton('📐', elem_id=f'{tabname}_send_to_extras', tooltip="Send image and generation parameters to extras tab.")
                }

            open_folder_button.click(
                fn=lambda: open_folder(shared.opts.outdir_samples or outdir),
                inputs=[],
                outputs=[],
            )

            if tabname != "extras":
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')

                with gr.Group():
                    html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
                    html_log = gr.HTML(elem_id=f'html_log_{tabname}', elem_classes="html-log")

                    generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                    if tabname == 'txt2img' or tabname == 'img2img':
                        generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                        generation_info_button.click(
                            fn=update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[generation_info, html_info, html_info],
                            outputs=[html_info, html_info],
                            show_progress=False,
                        )

                    save.click(
                        fn=call_queue.wrap_gradio_call(save_files),
                        _js="(x, y, z, w) => [x, y, false, selected_gallery_index()]",
                        inputs=[
                            generation_info,
                            result_gallery,
                            html_info,
                            html_info,
                        ],
                        outputs=[
                            download_files,
                            html_log,
                        ],
                        show_progress=False,
                    )

                    save_zip.click(
                        fn=call_queue.wrap_gradio_call(save_files),
                        _js="(x, y, z, w) => [x, y, true, selected_gallery_index()]",
                        inputs=[
                            generation_info,
                            result_gallery,
                            html_info,
                            html_info,
                        ],
                        outputs=[
                            download_files,
                            html_log,
                        ]
                    )

            else:
                html_info_x = gr.HTML(elem_id=f'html_info_x_{tabname}')
                html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                html_log = gr.HTML(elem_id=f'html_log_{tabname}')
                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")

                generation_info_button.click(fn=update_generation_info, _js="(x, y, z) => [x, y, selected_gallery_index()]", show_progress=False, # triggered on gallery change from js
                    inputs=[generation_info, html_info, html_info],
                    outputs=[html_info, html_info_formatted],
                )
                save.click(fn=call_queue.wrap_gradio_call(save_files), _js="(x, y, z, i) => [x, y, z, selected_gallery_index()]", show_progress=False,
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[download_files, html_log],
                )
                delete.click(fn=call_queue.wrap_gradio_call(delete_files), _js="(x, y, z, i) => [x, y, z, selected_gallery_index()]",
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[result_gallery, html_log],
                )

            if tabname == "txt2img":
                paste_field_names = modules.scripts.scripts_txt2img.paste_field_names
            elif tabname == "img2img":
                paste_field_names = modules.scripts.scripts_img2img.paste_field_names
            else:
                paste_field_names = []
            for paste_tabname, paste_button in buttons.items():
                debug(f'Create output panel: button={paste_button} tabname={paste_tabname}')
                bindings = parameters_copypaste.ParamBinding(paste_button=paste_button, tabname=paste_tabname, source_tabname=("txt2img" if tabname == "txt2img" else None), source_image_component=result_gallery, paste_field_names=paste_field_names)
                parameters_copypaste.register_paste_params_button(bindings)
            return result_gallery, generation_info, html_info, html_info_formatted, html_log

<<<<<<< HEAD

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id, visible: bool = True):
=======
            return result_gallery, generation_info if tabname != "extras" else html_info_x, html_info, html_log


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
<<<<<<< HEAD
        for k, v in args.items():
            setattr(refresh_component, k, v)
        return gr.update(**(args or {}))

    from modules.ui_components import ToolButton
    refresh_button = ToolButton(value=symbols.refresh, elem_id=elem_id, visible=visible)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component])
    return refresh_button

def create_browse_button(browse_component, elem_id):

    def browse(folder):
        # import subprocess
        if folder is not None:
            return gr.update(value = folder)
        return gr.update()

    from modules.ui_components import ToolButton
    browse_button = ToolButton(value=symbols.folder, elem_id=elem_id)
    browse_button.click(fn=browse, _js="async () => await browseFolder()", inputs=[browse_component], outputs=[browse_component])
    # browse_button.click(fn=browse, inputs=[browse_component], outputs=[browse_component])
    return browse_button
=======

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button


def setup_dialog(button_show, dialog, *, button_close=None):
    """Sets up the UI so that the dialog (gr.Box) is invisible, and is only shown when buttons_show is clicked, in a fullscreen modal window."""

    dialog.visible = False

    button_show.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[dialog],
    ).then(fn=None, _js="function(){ popupId('" + dialog.elem_id + "'); }")

    if button_close:
        button_close.click(fn=None, _js="closePopup")

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
