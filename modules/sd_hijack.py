from types import MethodType, SimpleNamespace
import torch
from torch.nn.functional import silu
<<<<<<< HEAD
=======
from types import MethodType

from modules import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet, patches
from modules.hypernetworks import hypernetwork
from modules.shared import cmd_opts
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr, xlmr_m18

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
import ldm.modules.attention
import ldm.modules.distributions.distributions
import ldm.modules.diffusionmodules.model
import ldm.modules.diffusionmodules.openaimodel
import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import ldm.modules.encoders.modules

<<<<<<< HEAD
import modules.textual_inversion.textual_inversion
from modules import devices, sd_hijack_optimizations, shared
from modules.hypernetworks import hypernetwork
from modules.shared import opts
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr
=======
import sgm.modules.attention
import sgm.modules.diffusionmodules.model
import sgm.modules.diffusionmodules.openaimodel
import sgm.modules.encoders.modules
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# silence new console spam from SD2
ldm.modules.attention.print = shared.ldm_print
ldm.modules.diffusionmodules.model.print = shared.ldm_print
ldm.util.print = shared.ldm_print
ldm.models.diffusion.ddpm.print = shared.ldm_print

optimizers = []
current_optimizer: sd_hijack_optimizations.SdOptimization = None

ldm_patched_forward = sd_unet.create_unet_forward(ldm.modules.diffusionmodules.openaimodel.UNetModel.forward)
ldm_original_forward = patches.patch(__file__, ldm.modules.diffusionmodules.openaimodel.UNetModel, "forward", ldm_patched_forward)

sgm_patched_forward = sd_unet.create_unet_forward(sgm.modules.diffusionmodules.openaimodel.UNetModel.forward)
sgm_original_forward = patches.patch(__file__, sgm.modules.diffusionmodules.openaimodel.UNetModel, "forward", sgm_patched_forward)

current_optimizer = SimpleNamespace(**{ "name": "none" })

def list_optimizers():
    new_optimizers = script_callbacks.list_optimizers_callback()

    new_optimizers = [x for x in new_optimizers if x.is_available()]

    new_optimizers = sorted(new_optimizers, key=lambda x: x.priority, reverse=True)

    optimizers.clear()
    optimizers.extend(new_optimizers)


def apply_optimizations(option=None):
    global current_optimizer

    undo_optimizations()
<<<<<<< HEAD
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th
    optimization_method = None
    can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(torch.nn.functional.scaled_dot_product_attention)
    if devices.device == torch.device("cpu"):
        if opts.cross_attention_optimization == "Scaled-Dot-Product":
            shared.log.warning("Cross-attention: Scaled dot product is not available on CPU")
            can_use_sdp = False
        if opts.cross_attention_optimization == "xFormers":
            shared.log.warning("Cross-attention: xFormers is not available on CPU")
            shared.xformers_available = False

    shared.log.info(f"Cross-attention: optimization={opts.cross_attention_optimization} options={opts.cross_attention_options}")
    if opts.cross_attention_optimization == "Disabled":
        optimization_method = 'none'
    if can_use_sdp and opts.cross_attention_optimization == "Scaled-Dot-Product" and 'SDP disable memory attention' in opts.cross_attention_options:
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.scaled_dot_product_no_mem_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sdp_no_mem_attnblock_forward
        optimization_method = 'sdp-no-mem'
    elif can_use_sdp and opts.cross_attention_optimization == "Scaled-Dot-Product":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.scaled_dot_product_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sdp_attnblock_forward
        optimization_method = 'sdp'
    if shared.xformers_available and opts.cross_attention_optimization == "xFormers":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.xformers_attnblock_forward
        optimization_method = 'xformers'
    if opts.cross_attention_optimization == "Sub-quadratic":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.sub_quad_attnblock_forward
        optimization_method = 'sub-quadratic'
    if opts.cross_attention_optimization == "Split attention":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_v1
        optimization_method = 'v1'
    if opts.cross_attention_optimization == "InvokeAI's":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward_invokeAI
        optimization_method = 'invokeai'
    if opts.cross_attention_optimization == "Doggettx's":
        ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward
        optimization_method = 'doggettx'
    current_optimizer.name = optimization_method
    return optimization_method
=======

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    sgm.modules.diffusionmodules.model.nonlinearity = silu
    sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)

    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer.name
    else:
        print("Disabling attention optimization")
        return ''
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def undo_optimizations():
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

    sgm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""
    pass # pylint: disable=unnecessary-pass


def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
<<<<<<< HEAD
    loss = sd_model._old_get_loss(pred, target, mean=False) # pylint: disable=protected-access
=======
    loss = sd_model._old_get_loss(pred, target, mean=False)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
<<<<<<< HEAD
        sd_model._custom_loss_weight = w # pylint: disable=protected-access
=======
        sd_model._custom_loss_weight = w
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss # pylint: disable=protected-access
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss # pylint: disable=protected-access
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class StableDiffusionModelHijack:
    fixes = None
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    def __init__(self):
<<<<<<< HEAD
        self.embedding_db.add_embedding_dir(opts.embeddings_dir)
=======
        import modules.textual_inversion.textual_inversion

        self.extra_generation_params = {}
        self.comments = []

        self.embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def apply_optimizations(self, option=None):
        try:
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            errors.display(e, "applying cross attention optimization")
            undo_optimizations()

    def convert_sdxl_to_ssd(self, m):
        """Converts an SDXL model to a Segmind Stable Diffusion model (see https://huggingface.co/segmind/SSD-1B)"""

        delattr(m.model.diffusion_model.middle_block, '1')
        delattr(m.model.diffusion_model.middle_block, '2')
        for i in ['9', '8', '7', '6', '5', '4']:
            delattr(m.model.diffusion_model.input_blocks[7][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.input_blocks[8][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[0][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[1][1].transformer_blocks, i)
        delattr(m.model.diffusion_model.output_blocks[4][1].transformer_blocks, '1')
        delattr(m.model.diffusion_model.output_blocks[5][1].transformer_blocks, '1')
        devices.torch_gc()

    def hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            text_cond_models = []

            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                typename = type(embedder).__name__
                if typename == 'FrozenOpenCLIPEmbedder':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenCLIPEmbedder':
                    model_embeddings = embedder.transformer.text_model.embeddings
                    model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenOpenCLIPEmbedder2':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self, textual_inversion_key='clip_g')
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])

            if len(text_cond_models) == 1:
                m.cond_stage_model = text_cond_models[0]
            else:
                m.cond_stage_model = conditioner

        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation or type(m.cond_stage_model) == xlmr_m18.BertSeriesModelWithTransformation:
            model_embeddings = m.cond_stage_model.roberta.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
            m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        apply_weighted_forward(m)
        if m.cond_stage_key == "edit":
            sd_hijack_unet.hijack_ddpm_edit()

<<<<<<< HEAD
        if opts.ipex_optimize and shared.backend == shared.Backend.ORIGINAL:
            try:
                import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
                m.model.training = False
                m.model = ipex.optimize(m.model, dtype=devices.dtype_unet, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
                shared.log.info("Applied IPEX Optimize.")
            except Exception as err:
                shared.log.warning(f"IPEX Optimize not supported: {err}")
=======
        self.apply_optimizations()
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        if (opts.cuda_compile or opts.cuda_compile_vae or opts.cuda_compile_upscaler) and shared.opts.cuda_compile_backend != 'none' and shared.backend == shared.Backend.ORIGINAL:
            try:
                import logging
                shared.log.info(f"Compiling pipeline={m.model.__class__.__name__} mode={opts.cuda_compile_backend}")
                import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
                log_level = logging.WARNING if opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
                if hasattr(torch, '_logging'):
                    torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
                torch._dynamo.config.verbose = opts.cuda_compile_verbose # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = opts.cuda_compile_errors # pylint: disable=protected-access
                torch.backends.cudnn.benchmark = True
                if opts.cuda_compile_backend == 'hidet':
                    import hidet # pylint: disable=import-error
                    hidet.torch.dynamo_config.use_tensor_core(True)
                    hidet.torch.dynamo_config.search_space(2)
                m.model = torch.compile(m.model, mode=opts.cuda_compile_mode, backend=opts.cuda_compile_backend, fullgraph=opts.cuda_compile_fullgraph, dynamic=False)
                shared.log.info("Model complilation done.")
            except Exception as err:
                shared.log.warning(f"Model compile not supported: {err}")
            finally:
                from installer import setup_logging
                setup_logging()

        self.optimization_method = apply_optimizations()
        self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

        import modules.models.diffusion.ddpm_edit

        if isinstance(m, ldm.models.diffusion.ddpm.LatentDiffusion):
            sd_unet.original_forward = ldm_original_forward
        elif isinstance(m, modules.models.diffusion.ddpm_edit.LatentDiffusion):
            sd_unet.original_forward = ldm_original_forward
        elif isinstance(m, sgm.models.diffusion.DiffusionEngine):
            sd_unet.original_forward = sgm_original_forward
        else:
            sd_unet.original_forward = None


    def undo_hijack(self, m):
<<<<<<< HEAD
        if not hasattr(m, 'cond_stage_model'):
            return # not ldm model
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            m.cond_stage_model = m.cond_stage_model.wrapped
=======
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                    embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped
                if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
                    embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped

            if hasattr(m, 'cond_stage_model'):
                delattr(m, 'cond_stage_model')

        elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped
        undo_optimizations()
        undo_weighted_forward(m)
        self.apply_circular(False)
        self.layers = None
        self.clip = None


    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return
        self.circular_enabled = enable
        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []
        self.extra_generation_params = {}

    def get_prompt_lengths(self, text):
        if self.clip is None:
<<<<<<< HEAD
            return 0, 0
=======
            return "-", "-"

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        _, token_count = self.clip.process_texts([text])
        return token_count, self.clip.get_target_prompt_token_count(token_count)

    def redo_hijack(self, m):
        self.undo_hijack(m)
        self.hijack(m)


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = devices.cond_cast_unet(vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != devices.device:
            attr = attr.to(device=devices.device, dtype=(torch.float32 if devices.device.type == 'mps' else None))

    setattr(self, name, attr)


ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer

# Ensure samping from Guassian for DDPM follows types
ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample = lambda self: self.mean.to(self.parameters.dtype) + self.std.to(self.parameters.dtype) * torch.randn(self.mean.shape, dtype=self.parameters.dtype).to(device=self.parameters.device)
