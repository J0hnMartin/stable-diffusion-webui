import os
import re
import sys
<<<<<<< HEAD
import time
from collections import namedtuple
=======
import inspect
from collections import namedtuple
from dataclasses import dataclass

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
import gradio as gr
from modules import paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer
from installer import log

<<<<<<< HEAD
=======
from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

AlwaysVisible = object()
time_component = {}
time_setup = {}


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


<<<<<<< HEAD
class Script:
    name = None
=======
@dataclass
class OnComponent:
    component: gr.blocks.Block


class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    filename = None
    args_from = None
    args_to = None
    alwayson = False
    is_txt2img = False
    is_img2img = False
<<<<<<< HEAD
    api_info = None
    group = None
=======
    tabname = None

    group = None
    """A gr.Group component that has all script's UI inside it."""

    create_group = True
    """If False, for alwayson scripts, a group component will not be created."""

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    infotext_fields = None
    paste_field_names = None
    section = None

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only = False
    """If true, the script setup will only be run in Gradio UI, not in API"""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        raise NotImplementedError

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        pass # pylint: disable=unnecessary-pass

    def show(self, is_img2img): # pylint: disable=unused-argument
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise
        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.
        Usually the processing is done by calling the processing.process_images function.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass # pylint: disable=unnecessary-pass

    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass


    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.
        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.
        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds
        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """
        pass # pylint: disable=unnecessary-pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """
        pass # pylint: disable=unnecessary-pass

    def on_before_component(self, callback, *, elem_id):
        """
        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.

        May be called in show() or ui() - but it may be too late in latter as some components may already be created.

        This function is an alternative to before_component in that it also cllows to run before a component is created, but
        it doesn't require to be called for every created component - just for the one you need.
        """
        if self.on_before_component_elem_id is None:
            self.on_before_component_elem_id = []

        self.on_before_component_elem_id.append((elem_id, callback))

    def on_after_component(self, callback, *, elem_id):
        """
        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.
        """
        if self.on_after_component_elem_id is None:
            self.on_after_component_elem_id = []

        self.on_after_component_elem_id.append((elem_id, callback))

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""
        need_tabname = self.show(True) == self.show(False)
<<<<<<< HEAD
        tabkind = 'img2img' if self.is_img2img else 'txt2txt'
=======
        tabkind = 'img2img' if self.is_img2img else 'txt2img'
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))
        return f'script_{tabname}{title}_{item_id}'

    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        pass


class ScriptBuiltinUI(Script):
    setup_for_ui_only = True

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabname = ('img2img' if self.is_img2img else 'txt2img') + "_" if need_tabname else ""

        return f'{tabname}{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path", "priority"])
scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])

def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependeencies or circular dependencies
    """

<<<<<<< HEAD
def list_scripts(scriptdirname, extension):
    tmp_list = []
    base = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(base):
        for filename in sorted(os.listdir(base)):
            tmp_list.append(ScriptFile(paths.script_path, filename, os.path.join(base, filename), '50'))
    for ext in extensions.active():
        tmp_list += ext.list_files(scriptdirname, extension)
    priority_list = []
    for script in tmp_list:
        if os.path.splitext(script.path)[1].lower() == extension and os.path.isfile(script.path):
            if script.basedir == paths.script_path:
                priority = '0'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'scripts')):
                priority = '1'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions-builtin')):
                priority = '2'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions')):
                priority = '3'
            else:
                priority = '9'
            if os.path.isfile(os.path.join(base, "..", ".priority")):
                with open(os.path.join(base, "..", ".priority"), "r", encoding="utf-8") as f:
                    priority = priority + str(f.read().strip())
                    log.debug(f'Script priority override: ${script.name}:{priority}')
            else:
                priority = priority + script.priority
            priority_list.append(ScriptFile(script.basedir, script.filename, script.path, priority))
            # log.debug(f'Adding script: {script.basedir} {script.filename} {script.path} {priority}')
    priority_sort = sorted(priority_list, key=lambda item: item.priority + item.path.lower(), reverse=False)
    return priority_sort
=======
    visited = {}
    result = []

    def inner(name):
        visited[name] = True

        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


@dataclass
class ScriptWithDependencies:
    script_canonical_name: str
    file: ScriptFile
    requires: list
    load_before: list
    load_after: list


def list_scripts(scriptdirname, extension, *, include_extensions=True):
    scripts = {}

    loaded_extensions = {ext.canonical_name: ext for ext in extensions.active()}
    loaded_extensions_scripts = {ext.canonical_name: [] for ext in extensions.active()}

    # build script dependency map
    root_script_basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(root_script_basedir):
        for filename in sorted(os.listdir(root_script_basedir)):
            if not os.path.isfile(os.path.join(root_script_basedir, filename)):
                continue

            if os.path.splitext(filename)[1].lower() != extension:
                continue

            script_file = ScriptFile(paths.script_path, filename, os.path.join(root_script_basedir, filename))
            scripts[filename] = ScriptWithDependencies(filename, script_file, [], [], [])

    if include_extensions:
        for ext in extensions.active():
            extension_scripts_list = ext.list_files(scriptdirname, extension)
            for extension_script in extension_scripts_list:
                if not os.path.isfile(extension_script.path):
                    continue

                script_canonical_name = ("builtin/" if ext.is_builtin else "") + ext.canonical_name + "/" + extension_script.filename
                relative_path = scriptdirname + "/" + extension_script.filename

                script = ScriptWithDependencies(
                    script_canonical_name=script_canonical_name,
                    file=extension_script,
                    requires=ext.metadata.get_script_requirements("Requires", relative_path, scriptdirname),
                    load_before=ext.metadata.get_script_requirements("Before", relative_path, scriptdirname),
                    load_after=ext.metadata.get_script_requirements("After", relative_path, scriptdirname),
                )

                scripts[script_canonical_name] = script
                loaded_extensions_scripts[ext.canonical_name].append(script)

    for script_canonical_name, script in scripts.items():
        # load before requires inverse dependency
        # in this case, append the script name into the load_after list of the specified script
        for load_before in script.load_before:
            # if this requires an individual script to be loaded before
            other_script = scripts.get(load_before)
            if other_script:
                other_script.load_after.append(script_canonical_name)

            # if this requires an extension
            other_extension_scripts = loaded_extensions_scripts.get(load_before)
            if other_extension_scripts:
                for other_script in other_extension_scripts:
                    other_script.load_after.append(script_canonical_name)

        # if After mentions an extension, remove it and instead add all of its scripts
        for load_after in list(script.load_after):
            if load_after not in scripts and load_after in loaded_extensions_scripts:
                script.load_after.remove(load_after)

                for other_script in loaded_extensions_scripts.get(load_after, []):
                    script.load_after.append(other_script.script_canonical_name)

    dependencies = {}

    for script_canonical_name, script in scripts.items():
        for required_script in script.requires:
            if required_script not in scripts and required_script not in loaded_extensions:
                errors.report(f'Script "{script_canonical_name}" requires "{required_script}" to be loaded, but it is not.', exc_info=False)

        dependencies[script_canonical_name] = script.load_after

    ordered_scripts = topological_sort(dependencies)
    scripts_list = [scripts[script_canonical_name].file for script_canonical_name in ordered_scripts]

    return scripts_list
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def list_files_with_name(filename):
    res = []
    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]
    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue
        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)
    return res


def load_scripts():
    t = timer.Timer()
    t0 = time.time()
    global current_basedir # pylint: disable=global-statement
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()
<<<<<<< HEAD
    scripts_list = list_scripts("scripts", ".py")
    syspath = sys.path

    def register_scripts_from_module(module, scriptfile):
        for script_class in module.__dict__.values():
            if type(script_class) != type:
=======

    scripts_list = list_scripts("scripts", ".py") + list_scripts("modules/processing_scripts", ".py", include_extensions=False)

    syspath = sys.path

    def register_scripts_from_module(module):
        for script_class in module.__dict__.values():
            if not inspect.isclass(script_class):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
                continue
            # log.debug(f'Registering script: {scriptfile.path}')
            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

<<<<<<< HEAD
    from installer import args
=======
    # here the scripts_list is already ordered
    # processing_script is not considered though
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    for scriptfile in scripts_list:
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir
            script_module = script_loading.load_module(scriptfile.path)
<<<<<<< HEAD
            register_scripts_from_module(script_module, scriptfile)
        except Exception as e:
            errors.display(e, f'Load script: {scriptfile.filename}')
=======
            register_scripts_from_module(script_module)

        except Exception:
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        finally:
            current_basedir = paths.script_path
<<<<<<< HEAD
            t.record(os.path.basename(scriptfile.basedir))
            sys.path = syspath
    global scripts_txt2img, scripts_img2img, scripts_postproc # pylint: disable=global-statement
    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
    return t, time.time()-t0
=======
            timer.startup_timer.record(scriptfile.filename)

    global scripts_txt2img, scripts_img2img, scripts_postproc

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
<<<<<<< HEAD
        res = func(*args, **kwargs)
        return res
    except Exception as e:
        errors.display(e, f'Calling script: {filename}/{funcname}')
=======
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    return default


class ScriptSummary:
    def __init__(self, op):
        self.start = time.time()
        self.update = time.time()
        self.op = op
        self.time = {}

    def record(self, script):
        self.update = time.time()
        self.time[script] = round(time.time() - self.update, 2)

    def report(self):
        total = sum(self.time.values())
        if total == 0:
            return
        scripts = [f'{k}:{v}' for k, v in self.time.items() if v > 0]
        log.debug(f'Script: op={self.op} total={total} scripts={scripts}')


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}
        self.infotext_fields = []
        self.paste_field_names = []
<<<<<<< HEAD
        self.script_load_ctr = 0
        self.is_img2img = False
        self.inputs = [None]
=======
        self.inputs = [None]

        self.on_before_component_elem_id = {}
        """dict of callbacks to be called before an element is created; key=elem_id, value=list of callbacks"""

        self.on_after_component_elem_id = {}
        """dict of callbacks to be called after an element is created; key=elem_id, value=list of callbacks"""
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.selectable_scripts.clear()
        self.alwayson_scripts.clear()
        self.titles.clear()
        self.infotext_fields.clear()
        self.paste_field_names.clear()
        self.script_load_ctr = 0
        self.is_img2img = is_img2img
        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()
        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

<<<<<<< HEAD
        for script_class, path, _basedir, _script_module in auto_processing_scripts + scripts_data:
            try:
                script = script_class()
                script.filename = path
                script.is_txt2img = not is_img2img
                script.is_img2img = is_img2img
                visibility = script.show(script.is_img2img)
                if visibility == AlwaysVisible:
                    self.scripts.append(script)
                    self.alwayson_scripts.append(script)
                    script.alwayson = True
                elif visibility:
                    self.scripts.append(script)
                    self.selectable_scripts.append(script)
            except Exception as e:
                log.error(f'Script initialize: {path} {e}')
=======
        for script_data in auto_processing_scripts + scripts_data:
            script = script_data.script_class()
            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def create_script_ui(self, script):
        import modules.api.models as api_models
        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)
        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)
        if controls is None:
            return
        script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()
        api_args = []
        for control in controls:
            if not isinstance(control, gr.components.IOComponent):
                log.error(f'Invalid script control: "{script.filename}" control={control}')
                continue
            control.custom_script_source = os.path.basename(script.filename)
            arg_info = api_models.ScriptArg(label=control.label or "")
            for field in ("value", "minimum", "maximum", "step", "choices"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)
            api_args.append(arg_info)
        script.api_info = api_models.ScriptInfo(name=script.name, is_img2img=script.is_img2img, is_alwayson=script.alwayson, args=api_args)
        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields
        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names
        self.inputs += controls
        script.args_to = len(self.inputs)

    def setup_ui_for_section(self, section, scriptlist=None):
        if scriptlist is None:
            scriptlist = self.alwayson_scripts
        for script in scriptlist:
            if script.alwayson and script.section != section:
                continue
            if script.create_group:
                with gr.Group(visible=script.alwayson) as group:
                    self.create_script_ui(script)
                script.group = group
            else:
                self.create_script_ui(script)

    def prepare_ui(self):
        self.inputs = [None]

        self.apply_on_before_component_callbacks()

    def apply_on_before_component_callbacks(self):
        for script in self.scripts:
            on_before = script.on_before_component_elem_id or []
            on_after = script.on_after_component_elem_id or []

            for elem_id, callback in on_before:
                if elem_id not in self.on_before_component_elem_id:
                    self.on_before_component_elem_id[elem_id] = []

                self.on_before_component_elem_id[elem_id].append((callback, script))

            for elem_id, callback in on_after:
                if elem_id not in self.on_after_component_elem_id:
                    self.on_after_component_elem_id[elem_id] = []

                self.on_after_component_elem_id[elem_id].append((callback, script))

            on_before.clear()
            on_after.clear()

    def create_script_ui(self, script):

        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)

        try:
            self.create_script_ui_inner(script)
        except Exception:
            errors.report(f"Error creating UI for {script.name}: ", exc_info=True)

    def create_script_ui_inner(self, script):
        import modules.api.models as api_models

        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)

        if controls is None:
            return

        script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()

        api_args = []

        for control in controls:
            control.custom_script_source = os.path.basename(script.filename)

            arg_info = api_models.ScriptArg(label=control.label or "")

            for field in ("value", "minimum", "maximum", "step"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)

            choices = getattr(control, 'choices', None)  # as of gradio 3.41, some items in choices are strings, and some are tuples where the first elem is the string
            if choices is not None:
                arg_info.choices = [x[0] if isinstance(x, tuple) else x for x in choices]

            api_args.append(arg_info)

        script.api_info = api_models.ScriptInfo(
            name=script.name,
            is_img2img=script.is_img2img,
            is_alwayson=script.alwayson,
            args=api_args,
        )

        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields

        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names

        self.inputs += controls
        script.args_to = len(self.inputs)

    def setup_ui_for_section(self, section, scriptlist=None):
        if scriptlist is None:
            scriptlist = self.alwayson_scripts

        for script in scriptlist:
            if script.alwayson and script.section != section:
                continue

            if script.create_group:
                with gr.Group(visible=script.alwayson) as group:
                    self.create_script_ui(script)

                script.group = group
            else:
                self.create_script_ui(script)

    def prepare_ui(self):
        self.inputs = [None]

    def setup_ui(self):
<<<<<<< HEAD
        import modules.api.models as api_models
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]
        inputs = []
        inputs_alwayson = [True]

        def create_script_ui(script, inputs, inputs_alwayson): # TODO this is legacy implementation, see self.create_script_ui
            script.args_from = len(inputs)
            script.args_to = len(inputs)
            controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)
            if controls is None:
                return
            script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()
            api_args = []
            for control in controls:
                if not isinstance(control, gr.components.IOComponent):
                    log.error(f'Invalid script control: "{script.filename}" control={control}')
                    continue
                control.custom_script_source = os.path.basename(script.filename)
                arg_info = api_models.ScriptArg(label=control.label or "")
                for field in ("value", "minimum", "maximum", "step", "choices"):
                    v = getattr(control, field, None)
                    if v is not None:
                        setattr(arg_info, field, v)
                api_args.append(arg_info)

            script.api_info = api_models.ScriptInfo(
                name=script.name,
                is_img2img=script.is_img2img,
                is_alwayson=script.alwayson,
                args=api_args,
            )
            if script.infotext_fields is not None:
                self.infotext_fields += script.infotext_fields
            if script.paste_field_names is not None:
                self.paste_field_names += script.paste_field_names
            inputs += controls
            inputs_alwayson += [script.alwayson for _ in controls]
            script.args_to = len(inputs)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        inputs.insert(0, dropdown)
        for script in self.selectable_scripts:
            with gr.Group(visible=False) as group:
                t0 = time.time()
                create_script_ui(script, inputs, inputs_alwayson)
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)
                script.group = group
=======
        all_titles = [wrap_call(script.title, script.filename, "title") or script.filename for script in self.scripts]
        self.title_map = {title.lower(): script for title, script in zip(all_titles, self.scripts)}
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]

        self.setup_ui_for_section(None)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        self.inputs[0] = dropdown

        self.setup_ui_for_section(None, self.selectable_scripts)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

        def select_script(script_index):
            selected_script = self.selectable_scripts[script_index - 1] if script_index > 0 else None
            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            if title == 'None': # called when an initial value is set from ui-config.json to show script's UI components
                return
            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field
        dropdown.change(fn=select_script, inputs=[dropdown], outputs=[script.group for script in self.selectable_scripts])

<<<<<<< HEAD
=======
        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=[script.group for script in self.selectable_scripts]
        )

        self.script_load_ctr = 0

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        def onload_script_visibility(params):
            title = params.get('Script', None)
            if title:
                title_index = self.titles.index(title)
                visibility = title_index == self.script_load_ctr
                self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                return gr.update(visible=visibility)
            else:
                return gr.update(visible=False)

<<<<<<< HEAD
        # with gr.Group(elem_id='scripts_alwayson_img2img' if self.is_img2img else 'scripts_alwayson_txt2img'):
        with gr.Accordion(label="Extensions", elem_id='scripts_alwayson_img2img' if self.is_img2img else 'scripts_alwayson_txt2img'):
            for script in self.alwayson_scripts:
                t0 = time.time()
                elem_id = f'script_{"txt2img" if script.is_txt2img else "img2img"}_{script.title().lower().replace(" ", "_")}'
                with gr.Group(elem_id=elem_id, elem_classes=['extension-script']) as group:
                    create_script_ui(script, inputs, inputs_alwayson)
                script.group = group
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)

        self.infotext_fields.append( (dropdown, lambda x: gr.update(value=x.get('Script', 'None'))) )
        self.infotext_fields.extend( [(script.group, onload_script_visibility) for script in self.selectable_scripts] )
        return inputs
=======
        self.infotext_fields.append((dropdown, lambda x: gr.update(value=x.get('Script', 'None'))))
        self.infotext_fields.extend([(script.group, onload_script_visibility) for script in self.selectable_scripts])

        self.apply_on_before_component_callbacks()

        return self.inputs
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def run(self, p, *args):
        s = ScriptSummary('run')
        script_index = args[0]
        if script_index == 0:
            return None
        script = self.selectable_scripts[script_index-1]
        if script is None:
            return None
        parsed = p.per_script_args.get(script.title(), args[script.args_from:script.args_to])
        processed = script.run(p, *parsed)
        s.record(script.title())
        s.report()
        return processed

<<<<<<< HEAD
    def before_process(self, p, **kwargs):
        s = ScriptSummary('before-process')
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process(p, *script_args, **kwargs)
            except Exception as e:
                errors.display(e, f"Error running before process: {script.filename}")
            s.record(script.title())
        s.report()

    def process(self, p, **kwargs):
        s = ScriptSummary('process')
        for script in self.alwayson_scripts:
            try:
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.process(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script process: {script.filename}')
            s.record(script.title())
        s.report()
=======
    def before_process(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process(p, *script_args)
            except Exception:
                errors.report(f"Error running before_process: {script.filename}", exc_info=True)

    def process(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                errors.report(f"Error running process: {script.filename}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def before_process_batch(self, p, **kwargs):
        s = ScriptSummary('before-process-batch')
        for script in self.alwayson_scripts:
            try:
<<<<<<< HEAD
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.before_process_batch(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before process batch: {script.filename}')
            s.record(script.title())
        s.report()
=======
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running before_process_batch: {script.filename}", exc_info=True)

    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running after_extra_networks_activate: {script.filename}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def process_batch(self, p, **kwargs):
        s = ScriptSummary('process-batch')
        for script in self.alwayson_scripts:
            try:
<<<<<<< HEAD
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.process_batch(p, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script process batch: {script.filename}')
            s.record(script.title())
        s.report()
=======
                script_args = p.script_args[script.args_from:script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_batch: {script.filename}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def postprocess(self, p, processed):
        s = ScriptSummary('postprocess')
        for script in self.alwayson_scripts:
            try:
<<<<<<< HEAD
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess(p, processed, *args)
            except Exception as e:
                errors.display(e, f'Running script postprocess: {script.filename}')
            s.record(script.title())
        s.report()
=======
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(f"Error running postprocess: {script.filename}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def postprocess_batch(self, p, images, **kwargs):
        s = ScriptSummary('postprocess-batch')
        for script in self.alwayson_scripts:
            try:
<<<<<<< HEAD
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess_batch(p, *args, images=images, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before postprocess batch: {script.filename}')
            s.record(script.title())
        s.report()

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        s = ScriptSummary('postprocess-batch-list')
        for script in self.alwayson_scripts:
            try:
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess_batch_list(p, pp, *args, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before postprocess batch list: {script.filename}')
            s.record(script.title())
        s.report()
=======
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch: {script.filename}", exc_info=True)

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch_list: {script.filename}", exc_info=True)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        s = ScriptSummary('postprocess-image')
        for script in self.alwayson_scripts:
            try:
<<<<<<< HEAD
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess_image(p, pp, *args)
            except Exception as e:
                errors.display(e, f'Running script postprocess image: {script.filename}')
            s.record(script.title())
        s.report()

    def before_component(self, component, **kwargs):
        s = ScriptSummary('before-component')
        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script before component: {script.filename}')
            s.record(script.title())
        s.report()

    def after_component(self, component, **kwargs):
        s = ScriptSummary('after-component')
        for script in self.scripts:
            try:
                script.after_component(component, **kwargs)
            except Exception as e:
                errors.display(e, f'Running script after component: {script.filename}')
            s.record(script.title())
        s.report()
=======
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    def before_component(self, component, **kwargs):
        for callback, script in self.on_before_component_elem_id.get(kwargs.get("elem_id"), []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_before_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running before_component: {script.filename}", exc_info=True)

    def after_component(self, component, **kwargs):
        for callback, script in self.on_after_component_elem_id.get(component.elem_id, []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_after_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.after_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running after_component: {script.filename}", exc_info=True)

    def script(self, title):
        return self.title_map.get(title.lower())
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

    def reload_sources(self, cache):
        s = ScriptSummary('reload-sources')
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename
            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module
<<<<<<< HEAD
=======

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to
            s.record(script.title())
        s.report()

    def before_hr(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                errors.report(f"Error running before_hr: {script.filename}", exc_info=True)

<<<<<<< HEAD
scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
=======
    def setup_scrips(self, p, *, is_ui=True):
        for script in self.alwayson_scripts:
            if not is_ui and script.setup_for_ui_only:
                continue

            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.setup(p, *script_args)
            except Exception:
                errors.report(f"Error running setup: {script.filename}", exc_info=True)


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
scripts_current: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
reload_scripts = load_scripts  # compatibility alias


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


<<<<<<< HEAD
def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """
    elem_classes = []
    if hasattr(comp, "elem_classes"):
        elem_classes = comp.elem_classes
    if elem_classes is None:
        elem_classes = []
    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]
    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    if scripts_current is not None:
        scripts_current.before_component(self, **kwargs)
    script_callbacks.before_component_callback(self, **kwargs)
    res = original_IOComponent_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    script_callbacks.after_component_callback(self, **kwargs)
    if scripts_current is not None:
        scripts_current.after_component(self, **kwargs)
    return res


original_IOComponent_init = gr.components.IOComponent.__init__
gr.components.IOComponent.__init__ = IOComponent_init


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    return res


original_BlockContext_init = gr.blocks.BlockContext.__init__
gr.blocks.BlockContext.__init__ = BlockContext_init
=======
reload_scripts = load_scripts  # compatibility alias
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
