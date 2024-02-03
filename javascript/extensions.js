<<<<<<< HEAD
function extensions_apply(extensions_disabled_list, extensions_update_list, disable_all) {
  const disable = [];
  const update = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
    if (x.name.startsWith('update_') && x.checked) update.push(x.name.substring(7));
  });
  restartReload();
  log('Extensions apply:', { disable, update });
  return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

function extensions_check(info, extensions_disabled_list, search_text, sort_column) {
  const disable = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
  });
  const id = randomId();
  log('Extensions check:', { disable });
  return [id, JSON.stringify(disable), search_text, sort_column];
}

function install_extension(button, url) {
  button.disabled = 'disabled';
  button.value = 'Installing...';
  button.innerHTML = 'installing';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  log('Extension install:', { url });
  gradioApp().querySelector('#install_extension_button').click();
}

function uninstall_extension(button, url) {
  button.disabled = 'disabled';
  button.value = 'Uninstalling...';
  button.innerHTML = 'uninstalling';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  log('Extension uninstall:', { url });
  gradioApp().querySelector('#uninstall_extension_button').click();
}

function update_extension(button, url) {
  button.value = 'Updating...';
  button.innerHTML = 'updating';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  log('Extension update:', { url });
  gradioApp().querySelector('#update_extension_button').click();
=======

function extensions_apply(_disabled_list, _update_list, disable_all) {
    var disable = [];
    var update = [];

    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x) {
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }

        if (x.name.startsWith("update_") && x.checked) {
            update.push(x.name.substring(7));
        }
    });

    restart_reload();

    return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

function extensions_check() {
    var disable = [];

    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x) {
        if (x.name.startsWith("enable_") && !x.checked) {
            disable.push(x.name.substring(7));
        }
    });

    gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
        x.innerHTML = "Loading...";
    });


    var id = randomId();
    requestProgress(id, gradioApp().getElementById('extensions_installed_html'), null, function() {

    });

    return [id, JSON.stringify(disable)];
}

function install_extension_from_index(button, url) {
    button.disabled = "disabled";
    button.value = "Installing...";

    var textarea = gradioApp().querySelector('#extension_to_install textarea');
    textarea.value = url;
    updateInput(textarea);

    gradioApp().querySelector('#install_extension_button').click();
}

function config_state_confirm_restore(_, config_state_name, config_restore_type) {
    if (config_state_name == "Current") {
        return [false, config_state_name, config_restore_type];
    }
    let restored = "";
    if (config_restore_type == "extensions") {
        restored = "all saved extension versions";
    } else if (config_restore_type == "webui") {
        restored = "the webui version";
    } else {
        restored = "the webui version and all saved extension versions";
    }
    let confirmed = confirm("Are you sure you want to restore from this state?\nThis will reset " + restored + ".");
    if (confirmed) {
        restart_reload();
        gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x) {
            x.innerHTML = "Loading...";
        });
    }
    return [confirmed, config_state_name, config_restore_type];
}

function toggle_all_extensions(event) {
    gradioApp().querySelectorAll('#extensions .extension_toggle').forEach(function(checkbox_el) {
        checkbox_el.checked = event.target.checked;
    });
}

function toggle_extension() {
    let all_extensions_toggled = true;
    for (const checkbox_el of gradioApp().querySelectorAll('#extensions .extension_toggle')) {
        if (!checkbox_el.checked) {
            all_extensions_toggled = false;
            break;
        }
    }

    gradioApp().querySelector('#extensions .all_extensions_toggle').checked = all_extensions_toggled;
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
}
