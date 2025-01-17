import datetime
import json
import os

<<<<<<< HEAD:modules/textual_inversion/ti_logging.py
saved_params_shared = {"model_name", "model_hash", "initial_step", "num_of_dataset_images", "learn_rate", "batch_size", "clip_grad_mode", "clip_grad_value", "gradient_step", "data_root", "log_directory", "training_width", "training_height", "steps", "create_image_every", "template_file", "latent_sampling_method"}
saved_params_ti = {"embedding_name", "num_vectors_per_token", "save_embedding_every", "save_image_with_stored_embedding"}
saved_params_hypernet = {"hypernetwork_name", "layer_structure", "activation_func", "weight_init", "add_layer_norm", "use_dropout", "save_hypernetwork_every"}
=======
saved_params_shared = {
    "batch_size",
    "clip_grad_mode",
    "clip_grad_value",
    "create_image_every",
    "data_root",
    "gradient_step",
    "initial_step",
    "latent_sampling_method",
    "learn_rate",
    "log_directory",
    "model_hash",
    "model_name",
    "num_of_dataset_images",
    "steps",
    "template_file",
    "training_height",
    "training_width",
}
saved_params_ti = {
    "embedding_name",
    "num_vectors_per_token",
    "save_embedding_every",
    "save_image_with_stored_embedding",
}
saved_params_hypernet = {
    "activation_func",
    "add_layer_norm",
    "hypernetwork_name",
    "layer_structure",
    "save_hypernetwork_every",
    "use_dropout",
    "weight_init",
}
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e:modules/textual_inversion/logging.py
saved_params_all = saved_params_shared | saved_params_ti | saved_params_hypernet
saved_params_previews = {
    "preview_cfg_scale",
    "preview_height",
    "preview_negative_prompt",
    "preview_prompt",
    "preview_sampler_index",
    "preview_seed",
    "preview_steps",
    "preview_width",
}


def save_settings_to_file(log_directory, all_params):
    now = datetime.datetime.now()
    params = {"datetime": now.strftime("%Y-%m-%d %H:%M:%S")}
    keys = saved_params_all
    if all_params.get('preview_from_txt2img'):
        keys = keys | saved_params_previews
    params.update({k: v for k, v in all_params.items() if k in keys})
    filename = f"settings-{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(os.path.join(log_directory, filename), "w", encoding='utf-8') as file:
        print(f'Training settings file: {os.path.join(log_directory, filename)}')
        json.dump(params, file, indent=2)
