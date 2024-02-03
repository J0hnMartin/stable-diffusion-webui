<<<<<<< HEAD
const contextMenuInit = () => {
  let eventListenerApplied = false;
  const menuSpecs = new Map();

  const uid = () => Date.now().toString(36) + Math.random().toString(36).substring(2);

  function showContextMenu(event, element, menuEntries) {
    const posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    const posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    const oldMenu = gradioApp().querySelector('#context-menu');
    if (oldMenu) oldMenu.remove();
    const contextMenu = document.createElement('nav');
    contextMenu.id = 'context-menu';
    contextMenu.style.top = `${posy}px`;
    contextMenu.style.left = `${posx}px`;
    const contextMenuList = document.createElement('ul');
    contextMenuList.className = 'context-menu-items';
    contextMenu.append(contextMenuList);
    menuEntries.forEach((entry) => {
      const contextMenuEntry = document.createElement('a');
      contextMenuEntry.innerHTML = entry.name;
      contextMenuEntry.addEventListener('click', (e) => entry.func());
      contextMenuList.append(contextMenuEntry);
    });
    gradioApp().appendChild(contextMenu);
    const menuWidth = contextMenu.offsetWidth + 4;
    const menuHeight = contextMenu.offsetHeight + 4;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    if ((windowWidth - posx) < menuWidth) contextMenu.style.left = `${windowWidth - menuWidth}px`;
    if ((windowHeight - posy) < menuHeight) contextMenu.style.top = `${windowHeight - menuHeight}px`;
  }

  function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {
    let currentItems = menuSpecs.get(targetElementSelector);
    if (!currentItems) {
      currentItems = [];
      menuSpecs.set(targetElementSelector, currentItems);
    }
    const newItem = {
      id: `${targetElementSelector}_${uid()}`,
      name: entryName,
      func: entryFunction,
      isNew: true,
    };
    currentItems.push(newItem);
    return newItem.id;
  }

  function removeContextMenuOption(id) {
    menuSpecs.forEach((v, k) => {
      let index = -1;
      v.forEach((e, ei) => {
        if (e.id === id) { index = ei; }
      });
      if (index >= 0) v.splice(index, 1);
    });
  }

  async function addContextMenuEventListener() {
    if (eventListenerApplied) return;
    log('initContextMenu');
    gradioApp().addEventListener('click', (e) => {
      if (!e.isTrusted) return;
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
    });
    gradioApp().addEventListener('contextmenu', (e) => {
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        if (e.composedPath()[0].matches(k)) {
          showContextMenu(e, e.composedPath()[0], v);
          e.preventDefault();
        }
      });
    });
    eventListenerApplied = true;
  }
  return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

const initResponse = contextMenuInit();
const appendContextMenuOption = initResponse[0];
const removeContextMenuOption = initResponse[1];
const addContextMenuEventListener = initResponse[2];

function initContextMenu() {
  const generateForever = (genbuttonid, interruptbuttonid) => {
    if (window.generateOnRepeatInterval) {
      log('generateForever: cancel');
      clearInterval(window.generateOnRepeatInterval);
      window.generateOnRepeatInterval = null;
    } else {
      log('generateForever: start');
      const genbutton = gradioApp().querySelector(genbuttonid);
      const busy = document.getElementById('progressbar')?.style.display === 'block';
      if (!busy) genbutton.click();
      window.generateOnRepeatInterval = setInterval(() => {
        const pbBusy = document.getElementById('progressbar')?.style.display === 'block';
        if (!pbBusy) genbutton.click();
      }, 500);
    }
  };

  for (const tab of ['txt2img', 'img2img']) {
    for (const el of ['generate', 'interrupt', 'skip', 'pause', 'paste', 'clear_prompt', 'extra_networks_btn']) {
      const id = `#${tab}_${el}`;
      appendContextMenuOption(id, 'Copy to clipboard', () => navigator.clipboard.writeText(document.querySelector(`#${tab}_prompt > label > textarea`).value));
      appendContextMenuOption(id, 'Generate forever', () => generateForever(`#${tab}_generate`));
      appendContextMenuOption(id, 'Apply selected style', quickApplyStyle);
      appendContextMenuOption(id, 'Quick save style', quickSaveStyle);
      appendContextMenuOption(id, 'nVidia overlay', initNVML);
    }
  }
}

onUiLoaded(initContextMenu);
onAfterUiUpdate(() => addContextMenuEventListener());
=======

var contextMenuInit = function() {
    let eventListenerApplied = false;
    let menuSpecs = new Map();

    const uid = function() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };

    function showContextMenu(event, element, menuEntries) {
        let posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
        let posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;

        let oldMenu = gradioApp().querySelector('#context-menu');
        if (oldMenu) {
            oldMenu.remove();
        }

        let baseStyle = window.getComputedStyle(uiCurrentTab);

        const contextMenu = document.createElement('nav');
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = posy + 'px';
        contextMenu.style.left = posx + 'px';



        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        menuEntries.forEach(function(entry) {
            let contextMenuEntry = document.createElement('a');
            contextMenuEntry.innerHTML = entry['name'];
            contextMenuEntry.addEventListener("click", function() {
                entry['func']();
            });
            contextMenuList.append(contextMenuEntry);

        });

        gradioApp().appendChild(contextMenu);

        let menuWidth = contextMenu.offsetWidth + 4;
        let menuHeight = contextMenu.offsetHeight + 4;

        let windowWidth = window.innerWidth;
        let windowHeight = window.innerHeight;

        if ((windowWidth - posx) < menuWidth) {
            contextMenu.style.left = windowWidth - menuWidth + "px";
        }

        if ((windowHeight - posy) < menuHeight) {
            contextMenu.style.top = windowHeight - menuHeight + "px";
        }

    }

    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {

        var currentItems = menuSpecs.get(targetElementSelector);

        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        let newItem = {
            id: targetElementSelector + '_' + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true
        };

        currentItems.push(newItem);
        return newItem['id'];
    }

    function removeContextMenuOption(uid) {
        menuSpecs.forEach(function(v) {
            let index = -1;
            v.forEach(function(e, ei) {
                if (e['id'] == uid) {
                    index = ei;
                }
            });
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }
        gradioApp().addEventListener("click", function(e) {
            if (!e.isTrusted) {
                return;
            }

            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        gradioApp().addEventListener("contextmenu", function(e) {
            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
            menuSpecs.forEach(function(v, k) {
                if (e.composedPath()[0].matches(k)) {
                    showContextMenu(e, e.composedPath()[0], v);
                    e.preventDefault();
                }
            });
        });
        eventListenerApplied = true;

    }

    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

var initResponse = contextMenuInit();
var appendContextMenuOption = initResponse[0];
var removeContextMenuOption = initResponse[1];
var addContextMenuEventListener = initResponse[2];

(function() {
    //Start example Context Menu Items
    let generateOnRepeat = function(genbuttonid, interruptbuttonid) {
        let genbutton = gradioApp().querySelector(genbuttonid);
        let interruptbutton = gradioApp().querySelector(interruptbuttonid);
        if (!interruptbutton.offsetParent) {
            genbutton.click();
        }
        clearInterval(window.generateOnRepeatInterval);
        window.generateOnRepeatInterval = setInterval(function() {
            if (!interruptbutton.offsetParent) {
                genbutton.click();
            }
        },
        500);
    };

    let generateOnRepeat_txt2img = function() {
        generateOnRepeat('#txt2img_generate', '#txt2img_interrupt');
    };

    let generateOnRepeat_img2img = function() {
        generateOnRepeat('#img2img_generate', '#img2img_interrupt');
    };

    appendContextMenuOption('#txt2img_generate', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#txt2img_interrupt', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#img2img_generate', 'Generate forever', generateOnRepeat_img2img);
    appendContextMenuOption('#img2img_interrupt', 'Generate forever', generateOnRepeat_img2img);

    let cancelGenerateForever = function() {
        clearInterval(window.generateOnRepeatInterval);
    };

    appendContextMenuOption('#txt2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancelGenerateForever);

})();
//End example Context Menu Items

onAfterUiUpdate(addContextMenuEventListener);
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
