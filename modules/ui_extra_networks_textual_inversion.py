import os
<<<<<<< HEAD
import concurrent
from modules import shared, sd_hijack, sd_models, ui_extra_networks
from modules.textual_inversion.textual_inversion import Embedding
=======

from modules import ui_extra_networks, sd_hijack, shared
from modules.ui_extra_networks import quote_js
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Embedding')
        self.allow_negative_prompt = True
        self.embeddings = []

    def refresh(self):
        if sd_models.model_data.sd_model is None:
            return
        if shared.backend == shared.Backend.ORIGINAL:
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            sd_models.model_data.sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def create_item(self, embedding: Embedding):
        record = None
        try:
            path, _ext = os.path.splitext(embedding.filename)
            tags = {}
            if embedding.tag is not None:
                tags[embedding.tag]=1
            name = os.path.splitext(embedding.basename)[0]
            record = {
                "type": 'Embedding',
                "name": name,
                "filename": embedding.filename,
                "preview": self.find_preview(embedding.filename),
                "search_term": self.search_terms_from_path(name),
                "prompt": json.dumps(f" {os.path.splitext(embedding.name)[0]}"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "tags": tags,
                "mtime": os.path.getmtime(embedding.filename),
                "size": os.path.getsize(embedding.filename),
            }
            record["info"] = self.find_info(embedding.filename)
            record["description"] = self.find_description(embedding.filename, record["info"])
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=embedding file={embedding.filename} {e}")
        return record

    def create_item(self, name, index=None, enable_filter=True):
        embedding = sd_hijack.model_hijack.embedding_db.word_embeddings.get(name)
        if embedding is None:
            return

        path, ext = os.path.splitext(embedding.filename)
        return {
            "name": name,
            "filename": embedding.filename,
            "shorthash": embedding.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(embedding.filename) + " " + (embedding.hash or ""),
            "prompt": quote_js(embedding.name),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(embedding.filename)},
        }

    def list_items(self):
<<<<<<< HEAD

        def list_folder(folder):
            for filename in os.listdir(folder):
                fn = os.path.join(folder, filename)
                if os.path.isfile(fn) and (fn.lower().endswith(".pt") or fn.lower().endswith(".safetensors")):
                    embedding = Embedding(vec=0, name=os.path.basename(fn), filename=fn)
                    embedding.filename = fn
                    self.embeddings.append(embedding)
                elif os.path.isdir(fn) and not fn.startswith('.'):
                    list_folder(fn)

        if sd_models.model_data.sd_model is None:
            self.embeddings = []
            list_folder(shared.opts.embeddings_dir)
        elif shared.backend == shared.Backend.ORIGINAL:
            self.embeddings = list(sd_hijack.model_hijack.embedding_db.word_embeddings.values())
        elif hasattr(sd_models.model_data.sd_model, 'embedding_db'):
            self.embeddings = list(sd_models.model_data.sd_model.embedding_db.word_embeddings.values())
        else:
            self.embeddings = []
        self.embeddings = sorted(self.embeddings, key=lambda emb: emb.filename)

        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, net): net for net in self.embeddings}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    yield item
=======
        # instantiate a list to protect against concurrent modification
        names = list(sd_hijack.model_hijack.embedding_db.word_embeddings)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
