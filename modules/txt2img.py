<<<<<<< HEAD
import os
import modules.scripts
from modules import sd_samplers, shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
=======
from contextlib import closing

import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts
import modules.shared as shared
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
from modules.ui import plaintext_to_html
import gradio as gr


<<<<<<< HEAD
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def txt2img(id_task,
            prompt, negative_prompt, prompt_styles,
            steps, sampler_index, latent_index,
            full_quality, restore_faces, tiling,
            n_iter, batch_size,
            cfg_scale, image_cfg_scale, diffusers_guidance_rescale,
            clip_skip,
            seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
            height, width,
            enable_hr, denoising_strength,
            hr_scale, hr_upscaler, hr_force, hr_second_pass_steps, hr_resize_x, hr_resize_y,
            refiner_steps, refiner_start, refiner_prompt, refiner_negative,
            hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
            override_settings_texts,
            *args):

    debug(f'txt2img: id_task={id_task}|prompt={prompt}|negative={negative_prompt}|styles={prompt_styles}|steps={steps}|sampler_index={sampler_index}|latent_index={latent_index}|full_quality={full_quality}|restore_faces={restore_faces}|tiling={tiling}|batch_count={n_iter}|batch_size={batch_size}|cfg_scale={cfg_scale}|clip_skip={clip_skip}|seed={seed}|subseed={subseed}|subseed_strength={subseed_strength}|seed_resize_from_h={seed_resize_from_h}|seed_resize_from_w={seed_resize_from_w}|height={height}|width={width}|enable_hr={enable_hr}|denoising_strength={denoising_strength}|hr_scale={hr_scale}|hr_upscaler={hr_upscaler}|hr_force={hr_force}|hr_second_pass_steps={hr_second_pass_steps}|hr_resize_x={hr_resize_x}|hr_resize_y={hr_resize_y}|image_cfg_scale={image_cfg_scale}|diffusers_guidance_rescale={diffusers_guidance_rescale}|refiner_steps={refiner_steps}|refiner_start={refiner_start}|refiner_prompt={refiner_prompt}|refiner_negative={refiner_negative}|override_settings={override_settings_texts}')

    if shared.sd_model is None:
        shared.log.warning('Model not loaded')
        return [], '', '', 'Error: model not loaded'

    override_settings = create_override_settings_dict(override_settings_texts)
    if sampler_index is None:
        sampler_index = 0
    if latent_index is None:
        latent_index = 0

=======
def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, request: gr.Request, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
<<<<<<< HEAD
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=True,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        latent_sampler=sd_samplers.samplers[latent_index].name,
=======
        sampler_name=sampler_name,
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        clip_skip=clip_skip,
        width=width,
        height=height,
<<<<<<< HEAD
        full_quality=full_quality,
        restore_faces=restore_faces,
        tiling=tiling,
=======
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_force=hr_force,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
<<<<<<< HEAD
        refiner_steps=refiner_steps,
        refiner_start=refiner_start,
        refiner_prompt=refiner_prompt,
        refiner_negative=refiner_negative,
        hdr_clamp=hdr_clamp, hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold,
        hdr_center=hdr_center, hdr_channel_shift=hdr_channel_shift, hdr_full_shift=hdr_full_shift,
        hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundry=hdr_max_boundry,
=======
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        override_settings=override_settings,
    )
    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args
<<<<<<< HEAD
    processed = modules.scripts.scripts_txt2img.run(p, *args)
    if processed is None:
        processed = processing.process_images(p)
    p.close()
    if processed is None:
        return [], '', '', 'Error: processing failed'
    generation_info_js = processed.js()
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
=======

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
