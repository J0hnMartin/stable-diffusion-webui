<<<<<<< HEAD
=======
import sys

from modules.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
def postprocessing_scripts():
    import modules.scripts
    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae
    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae
    modules.sd_vae.refresh_vae_list()


<<<<<<< HEAD
def list_crossattention():
    return [
        "Disabled",
        "xFormers",
        "Scaled-Dot-Product",
        "Doggettx's",
        "InvokeAI's",
        "Sub-quadratic",
        "Split attention"
    ]

def get_pipelines():
    import diffusers
    from installer import log
    pipelines = { # note: not all pipelines can be used manually as they require prior pipeline next to decoder pipeline
        'Autodetect': None,
        'Stable Diffusion': getattr(diffusers, 'StableDiffusionPipeline', None),
        'Stable Diffusion Inpaint': getattr(diffusers, 'StableDiffusionInpaintPipeline', None),
        'Stable Diffusion Img2Img': getattr(diffusers, 'StableDiffusionImg2ImgPipeline', None),
        'Stable Diffusion Instruct': getattr(diffusers, 'StableDiffusionInstructPix2PixPipeline', None),
        'Stable Diffusion Upscale': getattr(diffusers, 'StableDiffusionUpscalePipeline', None),
        'Stable Diffusion XL': getattr(diffusers, 'StableDiffusionXLPipeline', None),
        'Stable Diffusion XL Img2Img': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
        'Stable Diffusion XL Inpaint': getattr(diffusers, 'StableDiffusionXLInpaintPipeline', None),
        'Stable Diffusion XL Instruct': getattr(diffusers, 'StableDiffusionXLInstructPix2PixPipeline', None),
        'Latent Consistency Model': getattr(diffusers, 'LatentConsistencyModelPipeline', None),
        'PixArt Alpha': getattr(diffusers, 'PixArtAlphaPipeline', None),
        'UniDiffuser': getattr(diffusers, 'UniDiffuserPipeline', None),
        'Wuerstchen': getattr(diffusers, 'WuerstchenCombinedPipeline', None),
        'Kandinsky 2.1': getattr(diffusers, 'KandinskyPipeline', None),
        'Kandinsky 2.2': getattr(diffusers, 'KandinskyV22Pipeline', None),
        'Kandinsky 3': getattr(diffusers, 'Kandinsky3Pipeline', None),
        'DeepFloyd IF': getattr(diffusers, 'IFPipeline', None),
        'Custom Diffusers Pipeline': getattr(diffusers, 'DiffusionPipeline', None),
        # Segmind SSD-1B, Segmind Tiny
    }
    for k, v in pipelines.items():
        if k != 'Autodetect' and v is None:
            log.error(f'Not available: pipeline={k} diffusers={diffusers.__version__} path={diffusers.__file__}')
    return pipelines
=======
def cross_attention_optimizations():
    import modules.sd_hijack

    return ["Automatic"] + [x.title() for x in modules.sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    import modules.sd_unet

    return ["Automatic"] + [x.label for x in modules.sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    import modules.sd_unet

    modules.sd_unet.list_unets()


def list_checkpoint_tiles(use_short=False):
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles(use_short)


def refresh_checkpoints():
    import modules.sd_models
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers
    return modules.sd_samplers.all_samplers


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    from modules import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


def get_infotext_names():
    from modules import generation_parameters_copypaste, shared
    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    for tab_data in generation_parameters_copypaste.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            if isinstance(name, str):
                res[name] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from modules import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import modules.sd_models

        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models

        modules.sd_models.model_data.set_sd_model(value)


sys.modules['modules.shared'].__class__ = Shared
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
