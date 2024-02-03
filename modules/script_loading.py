import io
import os
<<<<<<< HEAD
import contextlib
import importlib.util
import modules.errors as errors
from installer import setup_logging, args


preloaded = []
=======
import importlib.util

from modules import errors
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def load_module(path):
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(module_spec)
    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    try:
        if '/sd-extension-' in path: # safe extensions without stdout intercept
            module_spec.loader.exec_module(module)
        else:
            # stdout = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                module_spec.loader.exec_module(module)
            setup_logging() # reset since scripts can hijaack logging
            for line in stdout.getvalue().splitlines():
                if len(line) > 0:
                    errors.log.info(f"Extension: script='{os.path.relpath(path)}' {line.strip()}")
    except Exception as e:
        errors.display(e, f'Module load: {path}')
    if args.profile:
        errors.profile(pr, f'Scripts: {path}')
    return module


def preload_extensions(extensions_dir, parser, extension_list=None):
    if not os.path.isdir(extensions_dir):
        return
<<<<<<< HEAD
    for dirname in sorted(os.listdir(extensions_dir)):
        if dirname in preloaded:
            continue
        preloaded.append(dirname)
=======

    extensions = extension_list if extension_list is not None else os.listdir(extensions_dir)
    for dirname in sorted(extensions):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        if not os.path.isfile(preload_script):
            continue
        try:
            module = load_module(preload_script)
            if hasattr(module, 'preload'):
                module.preload(parser)
<<<<<<< HEAD
        except Exception as e:
            errors.display(e, f'Extension preload: {preload_script}')
=======

        except Exception:
            errors.report(f"Error running preload() for {preload_script}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
