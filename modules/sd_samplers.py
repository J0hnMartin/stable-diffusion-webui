<<<<<<< HEAD
import os
from modules import sd_samplers_compvis, sd_samplers_kdiffusion, sd_samplers_diffusers, shared
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image # pylint: disable=unused-import


debug = shared.log.trace if os.environ.get('SD_SAMPLER_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: SAMPLER')
all_samplers = []
all_samplers = []
all_samplers_map = {}
samplers = all_samplers
samplers_for_img2img = all_samplers
=======
from modules import sd_samplers_kdiffusion, sd_samplers_timesteps, shared

# imports for functions that previously were here and are used by other modules
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image  # noqa: F401

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers = []
samplers_for_img2img = []
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
samplers_map = {}
samplers_hidden = {}


<<<<<<< HEAD
def list_samplers(backend_name = shared.backend):
    global all_samplers # pylint: disable=global-statement
    global all_samplers_map # pylint: disable=global-statement
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    global samplers_map # pylint: disable=global-statement
    if backend_name == shared.Backend.ORIGINAL:
        all_samplers = [*sd_samplers_compvis.samplers_data_compvis, *sd_samplers_kdiffusion.samplers_data_k_diffusion]
    else:
        all_samplers = [*sd_samplers_diffusers.samplers_data_diffusers]
    all_samplers_map = {x.name: x for x in all_samplers}
    samplers = all_samplers
    samplers_for_img2img = all_samplers
    samplers_map = {}
    # shared.log.debug(f'Available samplers: {[x.name for x in all_samplers]}')


def find_sampler_config(name):
    if name is not None and name != 'None':
=======
def find_sampler_config(name):
    if name is not None:
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]
    return config

<<<<<<< HEAD

def visible_sampler_names():
    visible_samplers = [x for x in all_samplers if x.name in shared.opts.show_samplers] if len(shared.opts.show_samplers) > 0 else all_samplers
    return visible_samplers
=======
    return config


def create_sampler(name, model):
    config = find_sampler_config(name)

    assert config is not None, f'bad sampler name: {name}'

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def create_sampler(name, model):
    if name == 'Default' and hasattr(model, 'scheduler'):
        config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_')}
        shared.log.debug(f'Sampler default {type(model.scheduler).__name__}: {config}')
        return model.scheduler
    config = find_sampler_config(name)
    if config is None:
        shared.log.error(f'Attempting to use unknown sampler: {name}')
        config = all_samplers[0]
    if shared.backend == shared.Backend.ORIGINAL:
        sampler = config.constructor(model)
        sampler.config = config
        sampler.initialize(p=None)
        sampler.name = name
        shared.log.debug(f'Sampler: sampler="{sampler.name}" config={sampler.config.options}')
        return sampler
    elif shared.backend == shared.Backend.DIFFUSERS:
        sampler = config.constructor(model)
        if not hasattr(model, 'scheduler_config'):
            model.scheduler_config = sampler.sampler.config.copy()
        model.scheduler = sampler.sampler
        shared.log.debug(f'Sampler: sampler="{sampler.name}" config={sampler.config}')
        return sampler.sampler
    else:
        return None


def set_samplers():
<<<<<<< HEAD
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    samplers = visible_sampler_names()
    samplers_for_img2img = [x for x in samplers if x.name != "PLMS"]
=======
    global samplers, samplers_for_img2img, samplers_hidden

    samplers_hidden = set(shared.opts.hide_samplers)
    samplers = all_samplers
    samplers_for_img2img = all_samplers

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name
<<<<<<< HEAD
=======


def visible_sampler_names():
    return [x.name for x in samplers if x.name not in samplers_hidden]


set_samplers()
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
