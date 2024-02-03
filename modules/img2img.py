import os
<<<<<<< HEAD
import itertools # SBM Batch frames
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, UnidentifiedImageError
=======
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr

from modules import images as imgutil
from modules.generation_parameters_copypaste import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
import modules.scripts
from modules import sd_samplers, shared, processing, images
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui import plaintext_to_html
from modules.memstats import memory_stats


<<<<<<< HEAD
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def process_batch(p, input_files, input_dir, output_dir, inpaint_mask_dir, args):
    shared.log.debug(f'batch: {input_files}|{input_dir}|{output_dir}|{inpaint_mask_dir}')
    processing.fix_seed(p)
    if input_files is not None and len(input_files) > 0:
        image_files = [f.name for f in input_files]
    else:
        if not os.path.isdir(input_dir):
            shared.log.error(f"Process batch: directory not found: {input_dir}")
            return
        image_files = shared.listfiles(input_dir)
    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = len(inpaint_masks) > 0
    if is_inpaint_batch:
        shared.log.info(f"Process batch: inpaint batch masks={len(inpaint_masks)}")
    save_normally = output_dir == ''
    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally
    shared.state.job_count = len(image_files) * p.n_iter
    if shared.opts.batch_frame_mode: # SBM Frame mode is on, process each image in batch with same seed
        window_size = p.batch_size
        btcrept = 1
        p.seed = [p.seed] * window_size # SBM MONKEYPATCH: Need to change processing to support a fixed seed value.
        p.subseed = [p.subseed] * window_size # SBM MONKEYPATCH
        shared.log.info(f"Process batch: inputs={len(image_files)} parallel={window_size} outputs={p.n_iter} per input ")
    else: # SBM Frame mode is off, standard operation of repeating same images with sequential seed.
        window_size = 1
        btcrept = p.batch_size
        shared.log.info(f"Process batch: inputs={len(image_files)} outputs={p.n_iter * p.batch_size} per input")
    for i in range(0, len(image_files), window_size):
        if shared.state.skipped:
            shared.state.skipped = False
        if shared.state.interrupted:
            break
        batch_image_files = image_files[i:i+window_size]
        batch_images = []
        for image_file in batch_image_files:
            try:
                img = Image.open(image_file)
                if p.scale_by != 1:
                    p.width = int(img.width * p.scale_by)
                    p.height = int(img.height * p.scale_by)
            except UnidentifiedImageError as e:
                shared.log.error(f"Image error: {e}")
                continue
            img = ImageOps.exif_transpose(img)
            batch_images.append(img)
        batch_images = batch_images * btcrept # Standard mode sends the same image per batchsize.
        p.init_images = batch_images
=======
def process_batch(p, input_dir, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    output_dir = output_dir.strip()
    processing.fix_seed(p)

    images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    state.job_count = len(images) * p.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps
    override_settings = p.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        try:
            img = Image.open(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            p.width = int(img.width * scale_by)
            p.height = int(img.height * scale_by)

        p.init_images = [img] * p.batch_size
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        image_path = Path(image)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
<<<<<<< HEAD
            batch_mask_images = []
            for image_file in batch_image_files:
                mask_image_path = os.path.join(inpaint_mask_dir, os.path.basename(image_file))
                # if not found use first one ("same mask for all images" use-case)
                if mask_image_path not in inpaint_masks:
                    mask_image_path = inpaint_masks[0]
                mask_image = Image.open(mask_image_path)
                batch_mask_images.append(mask_image)
            batch_mask_images = batch_mask_images * btcrept
            p.image_mask = batch_mask_images

        batch_image_files = batch_image_files * btcrept # List used for naming later.
=======
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = Image.open(mask_image_path)
            p.image_mask = mask_image
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                    info_img = Image.open(info_img_path)
                geninfo, _ = imgutil.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
            except Exception:
                parsed_parameters = {}

            p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
            p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            p.seed = int(parsed_parameters.get("Seed", seed))
            p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            p.steps = int(parsed_parameters.get("Steps", steps))

            model_info = get_closet_checkpoint_match(parsed_parameters.get("Model hash", None))
            if model_info is not None:
                p.override_settings['sd_model_checkpoint'] = model_info.name
            elif sd_model_checkpoint_override:
                p.override_settings['sd_model_checkpoint'] = sd_model_checkpoint_override
            else:
                p.override_settings.pop("sd_model_checkpoint", None)

        if output_dir:
            p.outpath_samples = output_dir
            p.override_settings['save_to_dirs'] = False
            p.override_settings['save_images_replace_action'] = "Add number suffix"
            if p.n_iter > 1 or p.batch_size > 1:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}-[generation_number]'
            else:
                p.override_settings['samples_filename_pattern'] = f'{image_path.stem}'

        proc = modules.scripts.scripts_img2img.run(p, *args)

        if proc is None:
<<<<<<< HEAD
            proc = processing.process_images(p)
        for n, (image, image_file) in enumerate(itertools.zip_longest(proc.images,batch_image_files)):
            basename = ''
            if shared.opts.use_original_name_batch:
                forced_filename, ext = os.path.splitext(os.path.basename(image_file))
            else:
                forced_filename = None
                ext = shared.opts.samples_format
            if len(proc.images) > 1:
                basename = f'{n + i}' if shared.opts.batch_frame_mode else f'{n}'
            else:
                basename = ''
            if output_dir == '':
                output_dir = shared.opts.outdir_img2img_samples
            if not save_normally:
                os.makedirs(output_dir, exist_ok=True)
            geninfo, items = images.read_info_from_image(image)
            for k, v in items.items():
                image.info[k] = v
            images.save_image(image, path=output_dir, basename=basename, seed=None, prompt=None, extension=ext, info=geninfo, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=image.info, forced_filename=forced_filename)
        shared.log.debug(f'Processed: images={len(batch_image_files)} memory={memory_stats()} batch')


def img2img(id_task: str, mode: int,
            prompt, negative_prompt, prompt_styles,
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
            n_iter, batch_size,
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
            override_settings_texts,
            *args): # pylint: disable=unused-argument

    if shared.sd_model is None:
        shared.log.warning('Model not loaded')
        return [], '', '', 'Error: model not loaded'

    debug(f'img2img: id_task={id_task}|mode={mode}|prompt={prompt}|negative_prompt={negative_prompt}|prompt_styles={prompt_styles}|init_img={init_img}|sketch={sketch}|init_img_with_mask={init_img_with_mask}|inpaint_color_sketch={inpaint_color_sketch}|inpaint_color_sketch_orig={inpaint_color_sketch_orig}|init_img_inpaint={init_img_inpaint}|init_mask_inpaint={init_mask_inpaint}|steps={steps}|sampler_index={sampler_index}|latent_index={latent_index}|mask_blur={mask_blur}|mask_alpha={mask_alpha}|inpainting_fill={inpainting_fill}|full_quality={full_quality}|restore_faces={restore_faces}|tiling={tiling}|n_iter={n_iter}|batch_size={batch_size}|cfg_scale={cfg_scale}|image_cfg_scale={image_cfg_scale}|clip_skip={clip_skip}|denoising_strength={denoising_strength}|seed={seed}|subseed{subseed}|subseed_strength={subseed_strength}|seed_resize_from_h={seed_resize_from_h}|seed_resize_from_w={seed_resize_from_w}|selected_scale_tab={selected_scale_tab}|height={height}|width={width}|scale_by={scale_by}|resize_mode={resize_mode}|resize_name={resize_name}|inpaint_full_res={inpaint_full_res}|inpaint_full_res_padding={inpaint_full_res_padding}|inpainting_mask_invert={inpainting_mask_invert}|img2img_batch_files={img2img_batch_files}|img2img_batch_input_dir={img2img_batch_input_dir}|img2img_batch_output_dir={img2img_batch_output_dir}|img2img_batch_inpaint_mask_dir={img2img_batch_inpaint_mask_dir}|override_settings_texts={override_settings_texts}')

    if mode == 5:
        if img2img_batch_files is None or len(img2img_batch_files) == 0:
            shared.log.debug('Init bactch images not set')
        elif init_img:
            shared.log.debug('Init image not set')

    if sampler_index is None:
        sampler_index = 0
    if latent_index is None:
        latent_index = 0

=======
            p.override_settings.pop('save_images_replace_action', None)
            proc = process_images(p)

        if not discard_further_results and proc:
            if batch_results:
                batch_results.images.extend(proc.images)
                batch_results.infotexts.extend(proc.infotexts)
            else:
                batch_results = proc

            if 0 <= shared.opts.img2img_batch_show_results_limit < len(batch_results.images):
                discard_further_results = True
                batch_results.images = batch_results.images[:int(shared.opts.img2img_batch_show_results_limit)]
                batch_results.infotexts = batch_results.infotexts[:int(shared.opts.img2img_batch_show_results_limit)]

    return batch_results


def img2img(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_name: str, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, request: gr.Request, *args):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    override_settings = create_override_settings_dict(override_settings_texts)

    if mode == 0:  # img2img
<<<<<<< HEAD
        if init_img is None:
            return [], '', '', 'Error: init image not provided'
        image = init_img.convert("RGB")
        mask = None
    elif mode == 1:  # img2img sketch
        if sketch is None:
            return [], '', '', 'Error: sketch image not provided'
        image = sketch.convert("RGB")
        mask = None
    elif mode == 2:  # inpaint
        if init_img_with_mask is None:
            return [], '', '', 'Error: init image with mask not provided'
        image = init_img_with_mask["image"]
        mask = init_img_with_mask["mask"]
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        image = image.convert("RGB")
=======
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        image = sketch
        mask = None
    elif mode == 2:  # inpaint
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        mask = processing.create_binary_mask(mask)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    elif mode == 3:  # inpaint sketch
        if inpaint_color_sketch is None:
            return [], '', '', 'Error: color sketch image not provided'
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    elif mode == 4:  # inpaint upload mask
        if init_img_inpaint is None:
            return [], '', '', 'Error: inpaint image not provided'
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        shared.log.error(f'Image processing unknown mode: {mode}')
        image = None
        mask = None
    if image is not None:
        image = ImageOps.exif_transpose(image)
        if selected_scale_tab == 1 and resize_mode != 0:
            width = int(image.width * scale_by)
            height = int(image.height * scale_by)

<<<<<<< HEAD
    p = processing.StableDiffusionProcessingImg2Img(
=======
    if selected_scale_tab == 1 and not is_batch:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
<<<<<<< HEAD
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=True,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        latent_sampler=sd_samplers.samplers[latent_index].name,
=======
        sampler_name=sampler_name,
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        clip_skip=clip_skip,
        width=width,
        height=height,
<<<<<<< HEAD
        full_quality=full_quality,
        restore_faces=restore_faces,
        tiling=tiling,
=======
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        resize_name=resize_name,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        refiner_steps=refiner_steps,
        refiner_start=refiner_start,
        inpaint_full_res=inpaint_full_res != 0,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        hdr_clamp=hdr_clamp, hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold,
        hdr_center=hdr_center, hdr_channel_shift=hdr_channel_shift, hdr_full_shift=hdr_full_shift,
        hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundry=hdr_max_boundry,
        override_settings=override_settings,
    )
    if selected_scale_tab == 1 and resize_mode != 0:
        p.scale_by = scale_by
    p.scripts = modules.scripts.scripts_img2img
    p.script_args = args
<<<<<<< HEAD
    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur
        p.extra_generation_params["Mask alpha"] = mask_alpha
        p.extra_generation_params["Mask invert"] = inpainting_mask_invert
        p.extra_generation_params["Mask content"] = inpainting_fill
        p.extra_generation_params["Mask area"] = inpaint_full_res
        p.extra_generation_params["Mask padding"] = inpaint_full_res_padding
    p.is_batch = mode == 5
    if p.is_batch:
        process_batch(p, img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args)
        processed = processing.Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = processing.process_images(p)
    p.close()
    generation_info_js = processed.js()
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
=======

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur

    with closing(p):
        if is_batch:
            assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
            processed = process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

            if processed is None:
                processed = Processed(p, [], p.seed, "")
        else:
            processed = modules.scripts.scripts_img2img.run(p, *args)
            if processed is None:
                processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
