window.opts = {};
window.localization = {};
window.titles = {};
let tabSelected = '';
let txt2img_textarea;
let img2img_textarea;
const wait_time = 800;
const token_timeouts = {};
let uiLoaded = false;

<<<<<<< HEAD
function rememberGallerySelection(name) {
  // dummy
}

function set_theme(theme) {
  const gradioURL = window.location.href;
  if (!gradioURL.includes('?__theme=')) window.location.replace(`${gradioURL}?__theme=${theme}`);
}

function update_token_counter(button_id) {
  if (token_timeouts[button_id]) clearTimeout(token_timeouts[button_id]);
  token_timeouts[button_id] = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function clip_gallery_urls(gallery) {
  const files = gallery.map((v) => v.data);
  navigator.clipboard.writeText(JSON.stringify(files)).then(
    () => log('clipboard:', files),
    (err) => console.error('clipboard:', files, err),
  );
}

function all_gallery_buttons() {
  const allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
  const visibleGalleryButtons = [];
  allGalleryButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleGalleryButtons.push(elem);
  });
  return visibleGalleryButtons;
}

function selected_gallery_button() {
  const allCurrentButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnail-item.thumbnail-small.selected');
  let visibleCurrentButton = null;
  allCurrentButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleCurrentButton = elem;
  });
  return visibleCurrentButton;
=======
function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }
    if (gallery.length == 1) {
        return [gallery[0]];
    }

    var index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [gallery[index]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    let buttons = gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains('selected')) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    var res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    let res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

function create_submit_args(args) {
    var res = Array.from(args);

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if (Array.isArray(res[res.length - 3])) {
        res[res.length - 3] = null;
    }

    return res;
}

function showSubmitButtons(tabname, show) {
    gradioApp().getElementById(tabname + '_interrupt').style.display = show ? "none" : "block";
    gradioApp().getElementById(tabname + '_skip').style.display = show ? "none" : "block";
}

function showRestoreProgressButton(tabname, show) {
    var button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;

    button.style.display = show ? "flex" : "none";
}

function submit() {
    showSubmitButtons('txt2img', false);

    var id = randomId();
    localSet("txt2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function submit_img2img() {
    showSubmitButtons('img2img', false);

    var id = randomId();
    localSet("img2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
        showSubmitButtons('img2img', true);
        localRemove("img2img_task_id");
        showRestoreProgressButton('img2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;
    res[1] = get_tab_index('mode_img2img');

    return res;
}

function submit_extras() {
    showSubmitButtons('extras', false);

    var id = randomId();

    requestProgress(id, gradioApp().getElementById('extras_gallery_container'), gradioApp().getElementById('extras_gallery'), function() {
        showSubmitButtons('extras', true);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    console.log(res);
    return res;
}

function restoreProgressTxt2img() {
    showRestoreProgressButton("txt2img", false);
    var id = localGet("txt2img_task_id");

    if (id) {
        requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
            showSubmitButtons('txt2img', true);
        }, null, 0);
    }

    return id;
}

function restoreProgressImg2img() {
    showRestoreProgressButton("img2img", false);

    var id = localGet("img2img_task_id");

    if (id) {
        requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
            showSubmitButtons('img2img', true);
        }, null, 0);
    }

    return id;
}


/**
 * Configure the width and height elements on `tabname` to accept
 * pasting of resolutions in the form of "width x height".
 */
function setupResolutionPasting(tabname) {
    var width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    var height = gradioApp().querySelector(`#${tabname}_height input[type=number]`);
    for (const el of [width, height]) {
        el.addEventListener('paste', function(event) {
            var pasteData = event.clipboardData.getData('text/plain');
            var parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (parsed) {
                width.value = parsed[1];
                height.value = parsed[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    }
}

onUiLoaded(function() {
    showRestoreProgressButton('txt2img', localGet("txt2img_task_id"));
    showRestoreProgressButton('img2img', localGet("img2img_task_id"));
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});


function modelmerger() {
    var id = randomId();
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function() {});

    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    var name_ = prompt('Style name:');
    return [name_, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}


var opts = {};
onAfterUiUpdate(function() {
    if (Object.keys(opts).length != 0) return;

    var json_elem = gradioApp().getElementById('settings_json');
    if (json_elem == null) return;

    var textarea = json_elem.querySelector('textarea');
    var jsdata = textarea.value;
    opts = JSON.parse(jsdata);

    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display = "none";

    setupTokenCounters();
});

onOptionsChanged(function() {
    var elem = gradioApp().getElementById('sd_checkpoint_hash');
    var sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    var shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://google.com/search?q=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea, img2img_textarea = undefined;

function restart_reload() {
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';

    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            location.reload();
        }, function() {
            setTimeout(requestPing, 500);
        });
    };

    setTimeout(requestPing, 2000);

    return [];
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
    let e = new Event("input", {bubbles: true});
    Object.defineProperty(e, "target", {value: target});
    target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click();
}

function currentImg2imgSourceResolution(w, h, scaleBy) {
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
    return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function() {
        gradioApp().getElementById('img2img_update_resize_to').click();
    }, 500);

    return [];

}



function setRandomSeed(elem_id) {
    var input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    var width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    var height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    if (!width || !height) return [];

    var tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}


var onEditTimers = {};

// calls func after afterMs milliseconds has passed since the input elem has beed enited by user
function onEdit(editId, elem, afterMs, func) {
    var edited = function() {
        var existingTimer = onEditTimers[editId];
        if (existingTimer) clearTimeout(existingTimer);

        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    elem.addEventListener("input", edited);

    return edited;
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
}

function selected_gallery_index() {
  const buttons = all_gallery_buttons();
  const button = selected_gallery_button();
  let result = -1;
  buttons.forEach((v, i) => { if (v === button) { result = i; } });
  return result;
}

function extract_image_from_gallery(gallery) {
  if (gallery.length === 0) return [null];
  if (gallery.length === 1) return [gallery[0]];
  let index = selected_gallery_index();
  if (index < 0 || index >= gallery.length) index = 0;
  return [gallery[index]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switchToTab(tab) {
  const tabs = Array.from(gradioApp().querySelectorAll('#tabs > .tab-nav > button'));
  const btn = tabs?.find((t) => t.innerText === tab);
  log('switchToTab', tab);
  if (btn) btn.click();
}

function switch_to_txt2img(...args) {
  switchToTab('Text');
  return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
  switchToTab('Image');
  gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}

function switch_to_img2img(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(0);
  return Array.from(arguments);
}

function switch_to_sketch(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(1);
  return Array.from(arguments);
}

function switch_to_inpaint(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(2);
  return Array.from(arguments);
}

function switch_to_inpaint_sketch(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(3);
  return Array.from(arguments);
}

function switch_to_extras(...args) {
  switchToTab('Process');
  return Array.from(arguments);
}

function switch_to_control(...args) {
  switchToTab('Control');
  return Array.from(arguments);
}

function get_tab_index(tabId) {
  let res = 0;
  gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button')
    .forEach((button, i) => {
      if (button.className.indexOf('selected') !== -1) res = i;
    });
  return res;
}

function create_tab_index_args(tabId, args) {
  const res = Array.from(args);
  res[0] = get_tab_index(tabId);
  return res;
}

function get_img2img_tab_index(...args) {
  const res = Array.from(arguments);
  res.splice(-2);
  res[0] = get_tab_index('mode_img2img');
  return res;
}

function create_submit_args(args) {
  const res = Array.from(args);
  // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
  // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
  // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
  // If gradio at some point stops sending outputs, this may break something
  if (Array.isArray(res[res.length - 3])) res[res.length - 3] = null;
  return res;
}

function showSubmitButtons(tabname, show) {}

function clearGallery(tabname) {
  const gallery = gradioApp().getElementById(`${tabname}_gallery`);
  gallery.classList.remove('logo');
  // gallery.style.height = window.innerHeight - gallery.getBoundingClientRect().top - 200 + 'px'
  const footer = gradioApp().getElementById(`${tabname}_footer`);
  footer.style.display = 'flex';
}

function submit_txt2img(...args) {
  log('submitTxt');
  clearGallery('txt2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('txt2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  return res;
}

function submit_img2img(...args) {
  log('submitImg');
  clearGallery('img2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('img2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = get_tab_index('mode_img2img');
  return res;
}

function submit_control(...args) {
  log('submitControl');
  clearGallery('control');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('control_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = gradioApp().querySelector('#control-tabs > .tab-nav > .selected')?.innerText.toLowerCase() || ''; // selected tab name
  return res;
}

function submit_postprocessing(...args) {
  log('SubmitExtras');
  clearGallery('extras');
  return args;
}

window.submit = submit_txt2img;

function modelmerger(...args) {
  const id = randomId();
  const res = create_submit_args(args);
  res[0] = id;
  return res;
}

function clearPrompts(prompt, negative_prompt) {
  prompt = '';
  negative_prompt = '';
  return [prompt, negative_prompt];
}

const promptTokecountUpdateFuncs = {};

function recalculatePromptTokens(name) {
  if (promptTokecountUpdateFuncs[name]) {
    promptTokecountUpdateFuncs[name]();
  }
}

function recalculate_prompts_txt2img(...args) {
  recalculatePromptTokens('txt2img_prompt');
  recalculatePromptTokens('txt2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_img2img(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_inpaint(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_control(...args) {
  recalculatePromptTokens('control_prompt');
  recalculatePromptTokens('control_neg_prompt');
  return Array.from(arguments);
}

function registerDragDrop() {
  const qs = gradioApp().getElementById('quicksettings');
  if (!qs) return;
  qs.addEventListener('dragover', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
  });
  qs.addEventListener('drop', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
    for (const f of evt.dataTransfer.files) {
      log('QuickSettingsDrop', f);
    }
  });
}

function sortUIElements() {
  // sort top-level tabs
  const currSelected = gradioApp()?.querySelector('.tab-nav > .selected')?.innerText;
  if (currSelected === tabSelected || !opts.ui_tab_reorder) return;
  tabSelected = currSelected;
  const tabs = gradioApp().getElementById('tabs')?.children[0];
  if (!tabs) return;
  let tabsOrder = opts.ui_tab_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];
  for (const el of Array.from(tabs.children)) {
    const elIndex = tabsOrder.indexOf(el.innerText.toLowerCase());
    if (elIndex > -1) el.style.order = elIndex - 50; // default is 0 so setting to negative values
  }
  // sort always-on scripts
  const find = (el, ordered) => {
    for (const i in ordered) {
      if (el.innerText.toLowerCase().startsWith(ordered[i])) return i;
    }
    return 99;
  };

  tabsOrder = opts.ui_scripts_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];

  const scriptsTxt = gradioApp().getElementById('scripts_alwayson_txt2img').children;
  for (const el of Array.from(scriptsTxt)) el.style.order = find(el, tabsOrder);

  const scriptsImg = gradioApp().getElementById('scripts_alwayson_img2img').children;
  for (const el of Array.from(scriptsImg)) el.style.order = find(el, tabsOrder);
  log('sortUIElements');
}

onAfterUiUpdate(async () => {
  let promptsInitialized = false;

  async function registerTextarea(id, id_counter, id_button) {
    const prompt = gradioApp().getElementById(id);
    if (!prompt) return;
    const counter = gradioApp().getElementById(id_counter);
    const localTextarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (counter.parentElement === prompt.parentElement) return;
    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = 'relative';
    promptTokecountUpdateFuncs[id] = () => { update_token_counter(id_button); };
    localTextarea.addEventListener('input', promptTokecountUpdateFuncs[id]);
    if (!promptsInitialized) log('initPrompts');
    promptsInitialized = true;
  }

  // sortUIElements();
  registerTextarea('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
  registerTextarea('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
  registerTextarea('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
  registerTextarea('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
  registerTextarea('control_prompt', 'control_token_counter', 'control_token_button');
  registerTextarea('control_neg_prompt', 'control_negative_token_counter', 'control_negative_token_button');
});

function update_txt2img_tokens(...args) {
  update_token_counter('txt2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function update_img2img_tokens(...args) {
  update_token_counter('img2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function getTranslation(...args) {
  return null;
}

function monitorServerStatus() {
  document.open();
  document.write(`
    <html>
      <head><title>SD.Next</title></head>
      <body style="background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center">
        <h1>Waiting for server...</h1>
        <script>
          function monitorServerStatus() {
            fetch('/sdapi/v1/progress')
              .then((res) => { !res?.ok ? setTimeout(monitorServerStatus, 1000) : location.reload(); })
              .catch((e) => setTimeout(monitorServerStatus, 1000))
          }
          window.onload = () => monitorServerStatus();
        </script>
      </body>
    </html>
  `);
  document.close();
}

function restartReload() {
  document.body.style = 'background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center';
  document.body.innerHTML = '<h1>Server shutdown in progress...</h1>';
  fetch('/sdapi/v1/progress')
    .then((res) => setTimeout(restartReload, 1000))
    .catch((e) => setTimeout(monitorServerStatus, 500));
  return [];
}

function updateInput(target) {
  const e = new Event('input', { bubbles: true });
  Object.defineProperty(e, 'target', { value: target });
  target.dispatchEvent(e);
}

let desiredCheckpointName = null;
function selectCheckpoint(name) {
  desiredCheckpointName = name;
  const tabname = getENActiveTab();
  const btnModel = gradioApp().getElementById(`${tabname}_extra_model`);
  const isRefiner = btnModel && btnModel.classList.contains('toolbutton-selected');
  if (isRefiner) gradioApp().getElementById('change_refiner').click();
  else gradioApp().getElementById('change_checkpoint').click();
  log(`Change ${isRefiner ? 'refiner' : 'model'}: ${desiredCheckpointName}`);
}

let desiredVAEName = null;
function selectVAE(name) {
  desiredVAEName = name;
  gradioApp().getElementById('change_vae').click();
  log(`Change VAE: ${desiredVAEName}`);
}

function selectReference(name) {
  desiredCheckpointName = name;
  gradioApp().getElementById('change_reference').click();
}

function currentImg2imgSourceResolution(_a, _b, scaleBy) {
  const img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
  return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
  setTimeout(() => gradioApp().getElementById('img2img_update_resize_to').click(), 500);
  return [];
}

function createThemeElement() {
  const el = document.createElement('img');
  el.id = 'theme-preview';
  el.className = 'theme-preview';
  el.onclick = () => { el.style.display = 'none'; };
  document.body.appendChild(el);
  return el;
}

function toggleCompact(val) {
  // log('toggleCompact', val);
  if (val) {
    gradioApp().style.setProperty('--layout-gap', 'var(--spacing-md)');
    gradioApp().querySelectorAll('input[type=range]').forEach((el) => el.classList.add('hidden'));
    gradioApp().querySelectorAll('div .form').forEach((el) => el.classList.add('form-compact'));
    gradioApp().querySelectorAll('.small-accordion .label-wrap').forEach((el) => el.classList.add('accordion-compact'));
  } else {
    gradioApp().style.setProperty('--layout-gap', 'var(--spacing-xxl)');
    gradioApp().querySelectorAll('input[type=range]').forEach((el) => el.classList.remove('hidden'));
    gradioApp().querySelectorAll('div .form').forEach((el) => el.classList.remove('form-compact'));
    gradioApp().querySelectorAll('.small-accordion .label-wrap').forEach((el) => el.classList.remove('accordion-compact'));
  }
}

function previewTheme() {
  let name = gradioApp().getElementById('setting_gradio_theme').querySelectorAll('input')?.[0].value || '';
  fetch('/file=html/themes.json').then((res) => {
    res.json().then((themes) => {
      const theme = themes.find((t) => t.id === name);
      if (theme) {
        window.open(theme.subdomain, '_blank');
      } else {
        const el = document.getElementById('theme-preview') || createThemeElement();
        el.style.display = el.style.display === 'block' ? 'none' : 'block';
        name = name.replace('/', '-');
        el.src = `/file=html/${name}.jpg`;
      }
    });
  });
}

async function browseFolder() {
  const f = await window.showDirectoryPicker();
  if (f && f.kind === 'directory') return f.name;
  return null;
}

async function reconnectUI() {
  const gallery = gradioApp().getElementById('txt2img_gallery');
  if (!gallery) return;
  const task_id = localStorage.getItem('task');
  const api_logo = Array.from(gradioApp().querySelectorAll('img')).filter((el) => el?.src?.endsWith('api-logo.svg'));
  if (api_logo.length > 0) api_logo[0].remove();
  clearInterval(start_check); // eslint-disable-line no-use-before-define
  if (task_id) {
    debug('task check:', task_id);
    requestProgress(task_id, null, gallery, null, null, true);
  }
  uiLoaded = true;

  const sd_model = gradioApp().getElementById('setting_sd_model_checkpoint');
  let loadingStarted = 0;
  let loadingMonitor = 0;

  const sd_model_callback = () => {
    const loading = sd_model.querySelector('.eta-bar');
    if (!loading) {
      loadingStarted = 0;
      clearInterval(loadingMonitor);
    } else if (loadingStarted === 0) {
      loadingStarted = Date.now();
      loadingMonitor = setInterval(() => {
        const elapsed = Date.now() - loadingStarted;
        if (elapsed > 3000 && loading) loading.style.display = 'none';
      }, 5000);
    }
  };
  const sd_model_observer = new MutationObserver(sd_model_callback);
  sd_model_observer.observe(sd_model, { attributes: true, childList: true, subtree: true });
  log('reconnectUI');
}

const start_check = setInterval(reconnectUI, 100);
