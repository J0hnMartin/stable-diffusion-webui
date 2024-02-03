import os
from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from modules.hashes import sha256_from_cache


class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetwork')

    def refresh(self):
        shared.reload_hypernetworks()

<<<<<<< HEAD
    def list_items(self):
        for name, path in shared.hypernetworks.items():
            try:
                name = os.path.relpath(os.path.splitext(path)[0], shared.opts.hypernetwork_dir)
                yield {
                    "type": 'Hypernetwork',
                    "name": name,
                    "filename": path,
                    "preview": self.find_preview(path),
                    "description": self.find_description(path),
                    "info": self.find_info(path),
                    "search_term": self.search_terms_from_path(name),
                    "prompt": json.dumps(f"<hypernet:{os.path.basename(name)}:{shared.opts.extra_networks_default_multiplier}>"),
                    "local_preview": f"{os.path.splitext(path)[0]}.{shared.opts.samples_format}",
                    "mtime": os.path.getmtime(path),
                    "size": os.path.getsize(path),
                }
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=hypernetwork file={path} {e}")
=======
    def create_item(self, name, index=None, enable_filter=True):
        full_path = shared.hypernetworks.get(name)
        if full_path is None:
            return

        path, ext = os.path.splitext(full_path)
        sha256 = sha256_from_cache(full_path, f'hypernet/{name}')
        shorthash = sha256[0:10] if sha256 else None

        return {
            "name": name,
            "filename": full_path,
            "shorthash": shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(path) + " " + (sha256 or ""),
            "prompt": quote_js(f"<hypernet:{name}:") + " + opts.extra_networks_default_multiplier + " + quote_js(">"),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(path + ext)},
        }

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(shared.hypernetworks)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def allowed_directories_for_previews(self):
        return [shared.opts.hypernetwork_dir]
