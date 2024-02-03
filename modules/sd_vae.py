import os
import collections
<<<<<<< HEAD
=======
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
import glob
from copy import deepcopy
import torch
from modules import shared, paths, devices, script_callbacks, sd_models


vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}
base_vae = None
loaded_vae_file = None
checkpoint_info = None
vae_path = os.path.abspath(os.path.join(paths.models_path, 'VAE'))
checkpoints_loaded = collections.OrderedDict()


<<<<<<< HEAD
=======
def get_loaded_vae_name():
    if loaded_vae_file is None:
        return None

    return os.path.basename(loaded_vae_file)


def get_loaded_vae_hash():
    if loaded_vae_file is None:
        return None

    sha256 = hashes.sha256(loaded_vae_file, 'vae')

    return sha256[0:10] if sha256 else None


>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info # pylint: disable=global-statement
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info # pylint: disable=global-statement
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file # pylint: disable=global-statement
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        shared.log.info("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    if filepath.endswith(".json"):
        return os.path.basename(os.path.dirname(filepath))
    else:
        return os.path.basename(filepath)


def refresh_vae_list():
    global vae_path # pylint: disable=global-statement
    vae_path = shared.opts.vae_dir
    vae_dict.clear()
    vae_paths = []
    if shared.backend == shared.Backend.ORIGINAL:
        if sd_models.model_path is not None and os.path.isdir(sd_models.model_path):
            vae_paths += [
                os.path.join(sd_models.model_path, 'VAE', '**/*.vae.ckpt'),
                os.path.join(sd_models.model_path, 'VAE', '**/*.vae.pt'),
                os.path.join(sd_models.model_path, 'VAE', '**/*.vae.safetensors'),
            ]
        if shared.opts.ckpt_dir is not None and os.path.isdir(shared.opts.ckpt_dir):
            vae_paths += [
                os.path.join(shared.opts.ckpt_dir, '**/*.vae.ckpt'),
                os.path.join(shared.opts.ckpt_dir, '**/*.vae.pt'),
                os.path.join(shared.opts.ckpt_dir, '**/*.vae.safetensors'),
            ]
        if shared.opts.vae_dir is not None and os.path.isdir(shared.opts.vae_dir):
            vae_paths += [
                os.path.join(shared.opts.vae_dir, '**/*.ckpt'),
                os.path.join(shared.opts.vae_dir, '**/*.pt'),
                os.path.join(shared.opts.vae_dir, '**/*.safetensors'),
            ]
    elif shared.backend == shared.Backend.DIFFUSERS:
        if sd_models.model_path is not None and os.path.isdir(sd_models.model_path):
            vae_paths += [os.path.join(sd_models.model_path, 'VAE', '**/*.vae.safetensors')]
        if shared.opts.ckpt_dir is not None and os.path.isdir(shared.opts.ckpt_dir):
            vae_paths += [os.path.join(shared.opts.ckpt_dir, '**/*.vae.safetensors')]
        if shared.opts.vae_dir is not None and os.path.isdir(shared.opts.vae_dir):
            vae_paths += [os.path.join(shared.opts.vae_dir, '**/*.safetensors')]
        vae_paths += [
            os.path.join(sd_models.model_path, 'VAE', '**/*.json'),
            os.path.join(shared.opts.vae_dir, '**/*.json'),
        ]
    candidates = []
    for path in vae_paths:
        candidates += glob.iglob(path, recursive=True)
    for filepath in candidates:
        name = get_filename(filepath)
        if name == 'VAE':
            continue
        if shared.backend == shared.Backend.ORIGINAL:
            vae_dict[name] = filepath
        else:
            if filepath.endswith(".json"):
                vae_dict[name] = os.path.dirname(filepath)
            else:
                vae_dict[name] = filepath
    shared.log.info(f'Available VAEs: path="{vae_path}" items={len(vae_dict)}')
    return vae_dict

    vae_dict.update(dict(sorted(vae_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_vae_near_checkpoint(checkpoint_file):
<<<<<<< HEAD
    checkpoint_path = os.path.splitext(checkpoint_file)[0]
    for vae_location in [f"{checkpoint_path}.vae.pt", f"{checkpoint_path}.vae.ckpt", f"{checkpoint_path}.vae.safetensors"]:
        if os.path.isfile(vae_location):
            return vae_location
    return None


def resolve_vae(checkpoint_file):
    if shared.opts.sd_vae == 'TAESD':
        return None, None
    if shared.cmd_opts.vae is not None: # 1st
        return shared.cmd_opts.vae, 'forced'
    if shared.opts.sd_vae == "None": # 2nd
        return None, None
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None: # 3rd
        return vae_near_checkpoint, 'near-checkpoint'
    if shared.opts.sd_vae == "Automatic": # 4th
        basename = os.path.splitext(os.path.basename(checkpoint_file))[0]
        if vae_dict.get(basename, None) is not None:
            return vae_dict[basename], 'automatic'
    else:
        vae_from_options = vae_dict.get(shared.opts.sd_vae, None) # 5th
        if vae_from_options is not None:
            return vae_from_options, 'settings'
        vae_from_options = vae_dict.get(shared.opts.sd_vae + '.safetensors', None) # 6th
        if vae_from_options is not None:
            return vae_from_options, 'settings'
        shared.log.warning(f"VAE not found: {shared.opts.sd_vae}")
    return None, None
=======
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    for vae_file in vae_dict.values():
        if os.path.basename(vae_file).startswith(checkpoint_path):
            return vae_file

    return None


@dataclass
class VaeResolution:
    vae: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.vae, self.source


def is_automatic():
    return shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_vae_from_setting() -> VaeResolution:
    if shared.opts.sd_vae == "None":
        return VaeResolution()

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return VaeResolution(vae_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")

    return VaeResolution(resolved=False)


def resolve_vae_from_user_metadata(checkpoint_file) -> VaeResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    vae_metadata = metadata.get("vae", None)
    if vae_metadata is not None and vae_metadata != "Automatic":
        if vae_metadata == "None":
            return VaeResolution()

        vae_from_metadata = vae_dict.get(vae_metadata, None)
        if vae_from_metadata is not None:
            return VaeResolution(vae_from_metadata, "from user metadata")

    return VaeResolution(resolved=False)


def resolve_vae_near_checkpoint(checkpoint_file) -> VaeResolution:
    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):
        return VaeResolution(vae_near_checkpoint, 'found near the checkpoint')

    return VaeResolution(resolved=False)


def resolve_vae(checkpoint_file) -> VaeResolution:
    if shared.cmd_opts.vae_path is not None:
        return VaeResolution(shared.cmd_opts.vae_path, 'from commandline argument')

    if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():
        return resolve_vae_from_setting()

    res = resolve_vae_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_vae_from_setting()

    return res
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def load_vae_dict(filename):
    vae_ckpt = sd_models.read_state_dict(filename)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


<<<<<<< HEAD
def load_vae(model, vae_file=None, vae_source="unknown-source"):
    global loaded_vae_file # pylint: disable=global-statement
=======
def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global vae_dict, base_vae, loaded_vae_file
    # save_settings = False

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0
    if vae_file:
        try:
            if cache_enabled and vae_file in checkpoints_loaded:
                # use vae checkpoint cache
                shared.log.info(f"Loading VAE: model={get_filename(vae_file)} source={vae_source} cached=True")
                store_base_vae(model)
                _load_vae_dict(model, checkpoints_loaded[vae_file])
            else:
                if not os.path.isfile(vae_file):
                    shared.log.error(f"VAE not found: model={vae_file} source={vae_source}")
                    return
                store_base_vae(model)
                vae_dict_1 = load_vae_dict(vae_file)
                _load_vae_dict(model, vae_dict_1)
                if cache_enabled:
                    # cache newly loaded vae
                    checkpoints_loaded[vae_file] = vae_dict_1.copy()
        except Exception as e:
            shared.log.error(f"Loading VAE failed: model={vae_file} source={vae_source} {e}")
            restore_base_vae(model)
        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU
        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file
    elif loaded_vae_file:
        restore_base_vae(model)
    loaded_vae_file = vae_file
    model.base_vae = base_vae
    model.loaded_vae_file = loaded_vae_file


def load_vae_diffusers(model_file, vae_file=None, vae_source="unknown-source"):
    if vae_file is None:
        return None
    if not os.path.exists(vae_file):
        shared.log.error(f'VAE not found: model{vae_file}')
        return None
    shared.log.info(f"Loading VAE: model={vae_file} source={vae_source}")
    diffusers_load_config = {
        "low_cpu_mem_usage": False,
        "torch_dtype": devices.dtype_vae,
        "use_safetensors": True,
    }
    if shared.opts.diffusers_vae_load_variant == 'default':
        if devices.dtype_vae == torch.float16:
            diffusers_load_config['variant'] = 'fp16'
    elif shared.opts.diffusers_vae_load_variant == 'fp32':
        pass
    else:
        diffusers_load_config['variant'] = shared.opts.diffusers_vae_load_variant
    if shared.opts.diffusers_vae_upcast != 'default':
        diffusers_load_config['force_upcast'] = True if shared.opts.diffusers_vae_upcast == 'true' else False
    shared.log.debug(f'Diffusers VAE load config: {diffusers_load_config}')
    try:
        import diffusers
        if os.path.isfile(vae_file):
            _pipeline, model_type = sd_models.detect_pipeline(model_file, 'vae')
            diffusers_load_config = { "config_file":  paths.sd_default_config if model_type != 'Stable Diffusion XL' else os.path.join(paths.sd_configs_path, 'sd_xl_base.yaml')}
            if os.path.getsize(vae_file) > 1310944880:
                vae = diffusers.ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder', **diffusers_load_config) # consistency decoder does not have from single file, so we'll just download it once more
            else:
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, **diffusers_load_config)
            vae = vae.to(devices.dtype_vae)
        else:
            if 'consistency-decoder' in vae_file:
                vae = diffusers.ConsistencyDecoderVAE.from_pretrained(vae_file, **diffusers_load_config)
            else:
                vae = diffusers.AutoencoderKL.from_pretrained(vae_file, **diffusers_load_config)
        global loaded_vae_file # pylint: disable=global-statement
        loaded_vae_file = os.path.basename(vae_file)
        # shared.log.debug(f'Diffusers VAE config: {vae.config}')
        return vae
    except Exception as e:
        shared.log.error(f"Loading VAE failed: model={vae_file} {e}")
    return None


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file # pylint: disable=global-statement
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
<<<<<<< HEAD
    from modules import lowvram, sd_hijack
=======
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    if not sd_model:
        sd_model = shared.sd_model
    if sd_model is None:
        return None
    global checkpoint_info # pylint: disable=global-statement
    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename
    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file).tuple()
    else:
        vae_source = "function-argument"
    if loaded_vae_file == vae_file:
        return None
    if not getattr(sd_model, 'has_accelerate', False):
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_model.to(devices.cpu)

<<<<<<< HEAD
    if shared.backend == shared.Backend.ORIGINAL:
        sd_hijack.model_hijack.undo_hijack(sd_model)
        if shared.cmd_opts.rollback_vae and devices.dtype_vae == torch.bfloat16:
            devices.dtype_vae = torch.float16
        load_vae(sd_model, vae_file, vae_source)
        sd_hijack.model_hijack.hijack(sd_model)
        script_callbacks.model_loaded_callback(sd_model)
        if vae_file is not None:
            shared.log.info(f"VAE weights loaded: {vae_file}")
=======
    if sd_model.lowvram:
        lowvram.send_everything_to_cpu()
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    else:
        if hasattr(shared.sd_model, "vae") and hasattr(shared.sd_model, "sd_checkpoint_info"):
            vae = load_vae_diffusers(shared.sd_model.sd_checkpoint_info.filename, vae_file, vae_source)
            if vae is not None:
                sd_models.set_diffuser_options(sd_model, vae=vae, op='vae')

<<<<<<< HEAD
    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram and not getattr(sd_model, 'has_accelerate', False):
=======
    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_vae(sd_model, vae_file, vae_source)

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not sd_model.lowvram:
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        sd_model.to(devices.device)
    return sd_model
