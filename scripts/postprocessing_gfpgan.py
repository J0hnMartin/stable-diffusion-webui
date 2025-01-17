<<<<<<< HEAD
from PIL import Image
import numpy as np
import gradio as gr
from modules import scripts_postprocessing
from modules.postprocess import gfpgan_model
from modules.ui_components import FormRow


class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    name = "GFPGAN"
    order = 2000

    def ui(self):
        with FormRow():
            gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Strength", value=0, elem_id="extras_gfpgan_visibility")
        return { "gfpgan_visibility": gfpgan_visibility }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, gfpgan_visibility): # pylint: disable=arguments-differ
        if gfpgan_visibility == 0:
            return
        restored_img = gfpgan_model.gfpgan_fix_faces(np.array(pp.image, dtype=np.uint8))
        res = Image.fromarray(restored_img)
        if gfpgan_visibility < 1.0:
            res = Image.blend(pp.image, res, gfpgan_visibility)
        pp.image = res
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
=======
from PIL import Image
import numpy as np

from modules import scripts_postprocessing, gfpgan_model, ui_components
import gradio as gr


class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    name = "GFPGAN"
    order = 2000

    def ui(self):
        with ui_components.InputAccordion(False, label="GFPGAN") as enable:
            gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id="extras_gfpgan_visibility")

        return {
            "enable": enable,
            "gfpgan_visibility": gfpgan_visibility,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, gfpgan_visibility):
        if gfpgan_visibility == 0 or not enable:
            return

        restored_img = gfpgan_model.gfpgan_fix_faces(np.array(pp.image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(pp.image, res, gfpgan_visibility)

        pp.image = res
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
