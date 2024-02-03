import json
<<<<<<< HEAD
import sys
import modules.errors as errors
=======
import os
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

from modules import errors, scripts

localizations = {}


def list_localizations(dirname): # pylint: disable=unused-argument
    localizations.clear()
    """
    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = [os.path.join(dirname, file)]

    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
<<<<<<< HEAD
        localizations[fn] = file.path
    """
    return localizations
=======
        if fn not in localizations:
            localizations[fn] = []
        localizations[fn].append(file.path)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def localization_js(current_localization_name: str) -> str:
    fns = localizations.get(current_localization_name, None)
    data = {}
<<<<<<< HEAD
    if fn is not None:
        try:
            with open(fn, "r", encoding="utf8") as file:
                data = json.load(file)
        except Exception as e:
            print(f"Error loading localization from {fn}:", file=sys.stderr)
            errors.display(e, 'localization')
=======
    if fns is not None:
        for fn in fns:
            try:
                with open(fn, "r", encoding="utf8") as file:
                    data.update(json.load(file))
            except Exception:
                errors.report(f"Error loading localization from {fn}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    return f"window.localization = {json.dumps(data)}"
