# no longer used, all paths are defined in paths.py

from modules.paths import modules_path, script_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, data_path, models_path, extensions_dir, extensions_builtin_dir # pylint: disable=unused-import

"""
import argparse
import os
import sys
import shlex

<<<<<<< HEAD
modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
=======
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)

cwd = os.getcwd()
modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
sd_configs_path = os.path.join(script_path, "configs")
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")

# Parse the --data-dir flag first so we can use it as a base for our other argument default values
parser_pre = argparse.ArgumentParser(add_help=False)
<<<<<<< HEAD
parser_pre.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
parser_pre.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
parser_pre.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", 'models'), help="Base path where all models are stored, default: %(default)s",)
=======
parser_pre.add_argument("--data-dir", type=str, default=os.path.dirname(modules_path), help="base path where all user data is stored", )
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
cmd_opts_pre = parser_pre.parse_known_args()[0]

# parser_pre.add_argument("--config", type=str, default=os.environ.get("SD_CONFIG", os.path.join(data_path, 'config.json')), help="Use specific server configuration file, default: %(default)s")

data_path = cmd_opts_pre.data_dir
models_path = cmd_opts_pre.models_dir if os.path.isabs(cmd_opts_pre.models_dir) else os.path.join(data_path, cmd_opts_pre.models_dir)
extensions_dir = os.path.join(data_path, "extensions")
<<<<<<< HEAD
extensions_builtin_dir = "extensions-builtin"

sd_model_file = cmd_opts_pre.ckpt or os.path.join(script_path, 'model.ckpt') # not used
default_sd_model_file = sd_model_file # not used
"""
=======
extensions_builtin_dir = os.path.join(script_path, "extensions-builtin")
config_states_dir = os.path.join(script_path, "config_states")

roboto_ttf_file = os.path.join(modules_path, 'Roboto-Regular.ttf')
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
