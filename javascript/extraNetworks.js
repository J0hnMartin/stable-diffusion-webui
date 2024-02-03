<<<<<<< HEAD
const activePromptTextarea = {};
let sortVal = 0;

// helpers

const requestGet = (url, data, handler) => {
  const xhr = new XMLHttpRequest();
  const args = Object.keys(data).map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(data[k])}`).join('&');
  xhr.open('GET', `${url}?${args}`, true);
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) handler(JSON.parse(xhr.responseText));
      else console.error(`Request: url=${url} status=${xhr.status} err`);
    }
  };
  xhr.send(JSON.stringify(data));
};

const getENActiveTab = () => {
  if (gradioApp().getElementById('tab_txt2img').style.display === 'block') return 'txt2img';
  if (gradioApp().getElementById('tab_img2img').style.display === 'block') return 'img2img';
  if (gradioApp().getElementById('tab_control').style.display === 'block') return 'control';
  return '';
};

const getENActivePage = () => {
  const tabname = getENActiveTab();
  const page = gradioApp().querySelector(`#${tabname}_extra_networks > .tabs > .tab-nav > .selected`);
  const pageName = page ? page.innerText : '';
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  if (btnApply) btnApply.style.display = pageName === 'Style' ? 'inline-flex' : 'none';
  return pageName;
};

const setENState = (state) => {
  if (!state) return;
  state.tab = getENActiveTab();
  state.page = getENActivePage();
  // log('setENState', state);
  const el = gradioApp().querySelector(`#${state.tab}_extra_state  > label > textarea`);
  el.value = JSON.stringify(state);
  updateInput(el);
};

// methods

function showCardDetails(event) {
  console.log('showCardDetails', event);
  const tabname = getENActiveTab();
  const btn = gradioApp().getElementById(`${tabname}_extra_details_btn`);
  btn.click();
  event.stopPropagation();
  event.preventDefault();
}

function getCardDetails(...args) {
  const el = event?.target?.parentElement?.parentElement;
  if (el?.classList?.contains('card')) setENState({ op: 'getCardDetails', item: el.dataset.name });
  else setENState({ op: 'getCardDetails', item: null });
  return [...args];
}

function readCardTags(el, tags) {
  const clickTag = (e, tag) => {
    e.preventDefault();
    e.stopPropagation();
    const textarea = activePromptTextarea[getENActiveTab()];
    if (textarea.value.indexOf(` ${tag}`) !== -1) textarea.value = textarea.value.replace(` ${tag}`, '');
    else if (textarea.value.indexOf(`${tag} `) !== -1) textarea.value = textarea.value.replace(` ${tag} `, '');
    else textarea.value += ` ${tag}`;
    updateInput(textarea);
  };
  if (tags.length === 0) return;
  const cardTags = tags.split('|');
  if (!cardTags || cardTags.length === 0) return;
  const tagsEl = el.getElementsByClassName('tags')[0];
  if (!tagsEl?.children || tagsEl.children.length > 0) return;
  for (const tag of cardTags) {
    const span = document.createElement('span');
    span.classList.add('tag');
    span.textContent = tag;
    span.onclick = (e) => clickTag(e, tag);
    tagsEl.appendChild(span);
  }
}

function readCardDescription(page, item) {
  requestGet('/sd_extra_networks/description', { page, item }, (data) => {
    const tabname = getENActiveTab();
    const description = gradioApp().querySelector(`#${tabname}_description > label > textarea`);
    description.value = data?.description?.trim() || '';
    // description.focus();
    updateInput(description);
    setENState({ op: 'readCardDescription', page, item });
  });
}

function getCardsForActivePage() {
  const pagename = getENActivePage();
  if (!pagename) return [];
  const allCards = Array.from(gradioApp().querySelectorAll('.extra-network-cards > .card'));
  const cards = allCards.filter((el) => el.dataset.page.toLowerCase().includes(pagename.toLowerCase()));
  log('getCardsForActivePage', pagename, cards.length);
  return allCards;
}

async function filterExtraNetworksForTab(searchTerm) {
  let found = 0;
  let items = 0;
  const t0 = performance.now();
  const pagename = getENActivePage();
  if (!pagename) return;
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);
    cards.forEach((elem) => {
      items += 1;
      if (searchTerm === '') {
        elem.style.display = '';
      } else {
        let text = elem.dataset.search.toLowerCase();
        text = text.toLowerCase().replace('models--', 'Diffusers').replace('\\', '/');
        if (text.indexOf(searchTerm) === -1) {
          elem.style.display = 'none';
        } else {
          elem.style.display = '';
          found += 1;
        }
      }
    });
  }
  const t1 = performance.now();
  if (found > 0) log(`filterExtraNetworks: text=${searchTerm} items=${items} match=${found} time=${Math.round(1000 * (t1 - t0)) / 1000000}`);
  else log(`filterExtraNetworks: text=all items=${items} time=${Math.round(1000 * (t1 - t0)) / 1000000}`);
}

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
  const re_extranet = /<([^:]+:[^:]+):[\d\.]+>/;
  const re_extranet_g = /\s+<([^:]+:[^:]+):[\d\.]+>/g;
  let m = text.match(re_extranet);
  let replaced = false;
  let newTextareaText;
  if (m) {
    const partToSearch = m[1];
    newTextareaText = textarea.value.replaceAll(re_extranet_g, (found) => {
      m = found.match(re_extranet);
      if (m[1] === partToSearch) {
        replaced = true;
        return '';
      }
      return found;
    });
  } else {
    newTextareaText = textarea.value.replaceAll(new RegExp(text, 'g'), (found) => {
      if (found === text) {
        replaced = true;
        return '';
      }
      return found;
    });
  }
  if (replaced) {
    textarea.value = newTextareaText;
    return true;
  }
  return false;
}

function sortExtraNetworks() {
  const sortDesc = ['Name [A-Z]', 'Name [Z-A]', 'Date [Newest]', 'Date [Oldest]', 'Size [Largest]', 'Size [Smallest]'];
  const pagename = getENActivePage();
  if (!pagename) return 'sort error: unknown page';
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  let num = 0;
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);
    num = cards.length;
    if (num === 0) return 'sort: no cards';
    cards.sort((a, b) => { // eslint-disable-line no-loop-func
      switch (sortVal) {
        case 0: return a.dataset.search ? a.dataset.search.localeCompare(b.dataset.search) : 0;
        case 1: return b.dataset.search ? b.dataset.search.localeCompare(a.dataset.search) : 0;
        case 2: return a.dataset.mtime && !isNaN(a.dataset.mtime) ? parseFloat(b.dataset.mtime) - parseFloat(a.dataset.mtime) : 0;
        case 3: return b.dataset.mtime && !isNaN(b.dataset.mtime) ? parseFloat(a.dataset.mtime) - parseFloat(b.dataset.mtime) : 0;
        case 4: return a.dataset.size && !isNaN(a.dataset.size) ? parseFloat(b.dataset.size) - parseFloat(a.dataset.size) : 0;
        case 5: return b.dataset.size && !isNaN(b.dataset.size) ? parseFloat(a.dataset.size) - parseFloat(b.dataset.size) : 0;
      }
      return 0;
    });
    for (const card of cards) pg.appendChild(card);
  }
  const desc = sortDesc[sortVal];
  sortVal = (sortVal + 1) % sortDesc.length;
  log('sortExtraNetworks', pagename, num, desc);
  return `sort page ${pagename} cards ${num} by ${desc}`;
}

function refreshENInput(tabname) {
  log('refreshExtraNetworks', tabname, gradioApp().querySelector(`#${tabname}_extra_networks textarea`)?.value);
  gradioApp().querySelector(`#${tabname}_extra_networks textarea`)?.dispatchEvent(new Event('input'));
}

function cardClicked(textToAdd, allowNegativePrompt) {
  const tabname = getENActiveTab();
  const textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
  if (textarea.value.indexOf(textToAdd) !== -1) textarea.value = textarea.value.replace(textToAdd, '');
  else textarea.value += textToAdd;
  updateInput(textarea);
}

function extraNetworksSearchButton(event) {
  const tabname = getENActiveTab();
  const searchTextarea = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const button = event.target;
  const text = button.classList.contains('search-all') ? '' : `${button.textContent.trim()}/`;
  searchTextarea.value = text;
  updateInput(searchTextarea);
}

let desiredStyle = '';
function selectStyle(name) {
  desiredStyle = name;
  const tabname = getENActiveTab();
  const button = gradioApp().querySelector(`#${tabname}_styles_select`);
  button.click();
}

function applyStyles(styles) {
  let newStyles = [];
  if (styles) newStyles = Array.isArray(styles) ? styles : [styles];
  const index = newStyles.indexOf(desiredStyle);
  if (index > -1) newStyles.splice(index, 1);
  else newStyles.push(desiredStyle);
  return newStyles.join('|');
}

function quickApplyStyle() {
  const tabname = getENActiveTab();
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  if (btnApply) btnApply.click();
}

function quickSaveStyle() {
  const tabname = getENActiveTab();
  const btnSave = gradioApp().getElementById(`${tabname}_extra_quicksave`);
  if (btnSave) btnSave.click();
}

let enDirty = false;
function closeDetailsEN(args) {
  // log('closeDetailsEN');
  enDirty = true;
  const tabname = getENActiveTab();
  const btnClose = gradioApp().getElementById(`${tabname}_extra_details_close`);
  if (btnClose) setTimeout(() => btnClose.click(), 100);
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  return args;
}

function refeshDetailsEN(args) {
  log(`refeshDetailsEN: ${enDirty}`);
  const tabname = getENActiveTab();
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  enDirty = false;
  return args;
}

// refresh on en show
function refreshENpage() {
  if (getCardsForActivePage().length === 0) {
    log('refreshENpage');
    const tabname = getENActiveTab();
    const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
    if (btnRefresh) btnRefresh.click();
  }
}

// init
function setupExtraNetworksForTab(tabname) {
  let tabs = gradioApp().querySelector(`#${tabname}_extra_tabs`);
  if (tabs) tabs.classList.add('extra-networks');
  const en = gradioApp().getElementById(`${tabname}_extra_networks`);
  tabs = gradioApp().querySelector(`#${tabname}_extra_tabs > div`);
  if (!tabs) return;

  // buttons
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  const btnScan = gradioApp().getElementById(`${tabname}_extra_scan`);
  const btnSave = gradioApp().getElementById(`${tabname}_extra_save`);
  const btnClose = gradioApp().getElementById(`${tabname}_extra_close`);
  const btnSort = gradioApp().getElementById(`${tabname}_extra_sort`);
  const btnView = gradioApp().getElementById(`${tabname}_extra_view`);
  const btnModel = gradioApp().getElementById(`${tabname}_extra_model`);
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  const buttons = document.createElement('span');
  buttons.classList.add('buttons');
  if (btnRefresh) buttons.appendChild(btnRefresh);
  if (btnModel) buttons.appendChild(btnModel);
  if (btnApply) buttons.appendChild(btnApply);
  if (btnScan) buttons.appendChild(btnScan);
  if (btnSave) buttons.appendChild(btnSave);
  if (btnSort) buttons.appendChild(btnSort);
  if (btnView) buttons.appendChild(btnView);
  if (btnClose) buttons.appendChild(btnClose);
  btnModel.onclick = () => btnModel.classList.toggle('toolbutton-selected');
  tabs.appendChild(buttons);

  // details
  const detailsImg = gradioApp().getElementById(`${tabname}_extra_details_img`);
  const detailsClose = gradioApp().getElementById(`${tabname}_extra_details_close`);
  if (detailsImg && detailsClose) {
    detailsImg.title = 'Close details';
    detailsImg.onclick = () => detailsClose.click();
  }

  // search and description
  const div = document.createElement('div');
  div.classList.add('second-line');
  tabs.appendChild(div);
  const txtSearch = gradioApp().querySelector(`#${tabname}_extra_search`);
  const txtSearchValue = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const txtDescription = gradioApp().getElementById(`${tabname}_description`);
  txtSearch.classList.add('search');
  txtDescription.classList.add('description');
  div.appendChild(txtSearch);
  div.appendChild(txtDescription);
  let searchTimer = null;
  txtSearchValue.addEventListener('input', (evt) => {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      filterExtraNetworksForTab(txtSearchValue.value.toLowerCase());
      searchTimer = null;
    }, 150);
  });

  // card hover
  let hoverTimer = null;
  let previousCard = null;
  gradioApp().getElementById(`${tabname}_extra_tabs`).onmouseover = (e) => {
    const el = e.target.closest('.card'); // bubble-up to card
    if (!el || (el.title === previousCard)) return;
    if (!hoverTimer) {
      hoverTimer = setTimeout(() => {
        readCardDescription(el.dataset.page, el.dataset.name);
        readCardTags(el, el.dataset.tags);
        previousCard = el.title;
      }, 300);
    }
    el.onmouseout = () => {
      clearTimeout(hoverTimer);
      hoverTimer = null;
    };
  };

  // en style
  if (!en) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    for (const el of Array.from(gradioApp().querySelectorAll('.extra-networks-page'))) {
      el.style.height = `${window.opts.extra_networks_height}vh`;
      el.parentElement.style.width = '-webkit-fill-available';
    }
    if (entries[0].intersectionRatio > 0) {
      refreshENpage();
      if (window.opts.extra_networks_card_cover === 'cover') {
        en.style.transition = '';
        en.style.zIndex = 100;
        en.style.position = 'absolute';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      } else if (window.opts.extra_networks_card_cover === 'sidebar') {
        en.style.transition = 'width 0.2s ease';
        en.style.zIndex = 100;
        en.style.position = 'absolute';
        en.style.right = '0';
        en.style.width = `${window.opts.extra_networks_sidebar_width}vw`;
        en.style.height = '-webkit-fill-available';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = `${100 - 2 - window.opts.extra_networks_sidebar_width}vw`;
      } else {
        en.style.transition = '';
        en.style.zIndex = 0;
        en.style.position = 'relative';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      }
    } else {
      en.style.width = 0;
      gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
    }
  });
  intersectionObserver.observe(en); // monitor visibility of
}

function setupExtraNetworks() {
  setupExtraNetworksForTab('txt2img');
  setupExtraNetworksForTab('img2img');
  setupExtraNetworksForTab('control');

  function registerPrompt(tabname, id) {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!textarea) return;
    if (!activePromptTextarea[tabname]) activePromptTextarea[tabname] = textarea;
    textarea.addEventListener('focus', () => { activePromptTextarea[tabname] = textarea; });
  }

  registerPrompt('txt2img', 'txt2img_prompt');
  registerPrompt('txt2img', 'txt2img_neg_prompt');
  registerPrompt('img2img', 'img2img_prompt');
  registerPrompt('img2img', 'img2img_neg_prompt');
  registerPrompt('control', 'control_prompt');
  registerPrompt('control', 'control_neg_prompt');
  log('initExtraNetworks');
}

onUiLoaded(setupExtraNetworks);
=======
function toggleCss(key, css, enable) {
    var style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function setupExtraNetworksForTab(tabname) {
    gradioApp().querySelector('#' + tabname + '_extra_tabs').classList.add('extra-networks');

    var tabs = gradioApp().querySelector('#' + tabname + '_extra_tabs > div');
    var searchDiv = gradioApp().getElementById(tabname + '_extra_search');
    var search = searchDiv.querySelector('textarea');
    var sort = gradioApp().getElementById(tabname + '_extra_sort');
    var sortOrder = gradioApp().getElementById(tabname + '_extra_sortorder');
    var refresh = gradioApp().getElementById(tabname + '_extra_refresh');
    var showDirsDiv = gradioApp().getElementById(tabname + '_extra_show_dirs');
    var showDirs = gradioApp().querySelector('#' + tabname + '_extra_show_dirs input');
    var promptContainer = gradioApp().querySelector('.prompt-container-compact#' + tabname + '_prompt_container');
    var negativePrompt = gradioApp().querySelector('#' + tabname + '_neg_prompt');

    tabs.appendChild(searchDiv);
    tabs.appendChild(sort);
    tabs.appendChild(sortOrder);
    tabs.appendChild(refresh);
    tabs.appendChild(showDirsDiv);

    var applyFilter = function() {
        var searchTerm = search.value.toLowerCase();

        gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
            var searchOnly = elem.querySelector('.search_only');
            var text = elem.querySelector('.name').textContent.toLowerCase() + " " + elem.querySelector('.search_term').textContent.toLowerCase();

            var visible = text.indexOf(searchTerm) != -1;

            if (searchOnly && searchTerm.length < 4) {
                visible = false;
            }

            elem.style.display = visible ? "" : "none";
        });

        applySort();
    };

    var applySort = function() {
        var cards = gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card');

        var reverse = sortOrder.classList.contains("sortReverse");
        var sortKey = sort.querySelector("input").value.toLowerCase().replace("sort", "").replaceAll(" ", "_").replace(/_+$/, "").trim() || "name";
        sortKey = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
        var sortKeyStore = sortKey + "-" + (reverse ? "Descending" : "Ascending") + "-" + cards.length;

        if (sortKeyStore == sort.dataset.sortkey) {
            return;
        }
        sort.dataset.sortkey = sortKeyStore;

        cards.forEach(function(card) {
            card.originalParentElement = card.parentElement;
        });
        var sortedCards = Array.from(cards);
        sortedCards.sort(function(cardA, cardB) {
            var a = cardA.dataset[sortKey];
            var b = cardB.dataset[sortKey];
            if (!isNaN(a) && !isNaN(b)) {
                return parseInt(a) - parseInt(b);
            }

            return (a < b ? -1 : (a > b ? 1 : 0));
        });
        if (reverse) {
            sortedCards.reverse();
        }
        cards.forEach(function(card) {
            card.remove();
        });
        sortedCards.forEach(function(card) {
            card.originalParentElement.appendChild(card);
        });
    };

    search.addEventListener("input", applyFilter);
    sortOrder.addEventListener("click", function() {
        sortOrder.classList.toggle("sortReverse");
        applySort();
    });
    applyFilter();

    extraNetworksApplySort[tabname] = applySort;
    extraNetworksApplyFilter[tabname] = applyFilter;

    var showDirsUpdate = function() {
        var css = '#' + tabname + '_extra_tabs .extra-network-subdirs { display: none; }';
        toggleCss(tabname + '_extra_show_dirs_style', css, !showDirs.checked);
        localSet('extra-networks-show-dirs', showDirs.checked ? 1 : 0);
    };
    showDirs.checked = localGet('extra-networks-show-dirs', 1) == 1;
    showDirs.addEventListener("change", showDirsUpdate);
    showDirsUpdate();
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector('.toprow-compact-tools')) return; // only applicable for compact prompt layout

    var promptContainer = gradioApp().getElementById(tabname + '_prompt_container');
    var prompt = gradioApp().getElementById(tabname + '_prompt_row');
    var negPrompt = gradioApp().getElementById(tabname + '_neg_prompt_row');
    var elem = id ? gradioApp().getElementById(id) : null;

    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    if (elem) {
        elem.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}


function extraNetworksUrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, '', false, false);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt) { // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);

}

function applyExtraNetworkFilter(tabname) {
    setTimeout(extraNetworksApplyFilter[tabname], 1);
}

function applyExtraNetworkSort(tabname) {
    setTimeout(extraNetworksApplySort[tabname], 1);
}

var extraNetworksApplyFilter = {};
var extraNetworksApplySort = {};
var activePromptTextarea = {};

function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');

    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    registerPrompt('txt2img', 'txt2img_prompt');
    registerPrompt('txt2img', 'txt2img_neg_prompt');
    registerPrompt('img2img', 'img2img_prompt');
    registerPrompt('img2img', 'img2img_neg_prompt');
}

onUiLoaded(setupExtraNetworks);

var re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
var re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
    var m = text.match(re_extranet);
    var replaced = false;
    var newTextareaText;
    if (m) {
        var extraTextBeforeNet = opts.extra_networks_add_text_separator;
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(re_extranet_g, function(found, net, pos) {
            m = found.match(re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });

        if (foundAtPosition >= 0) {
            if (newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition) + newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            if (newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) == extraTextBeforeNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) + newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        newTextareaText = textarea.value.replaceAll(new RegExp(text, "g"), function(found) {
            if (found == text) {
                replaced = true;
                return "";
            }
            return found;
        });
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

function cardClicked(tabname, textToAdd, allowNegativePrompt) {
    var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");

    if (!tryToRemoveExtraNetworkFromPrompt(textarea, textToAdd)) {
        textarea.value = textarea.value + opts.extra_networks_add_text_separator + textToAdd;
    }

    updateInput(textarea);
}

function saveCardPreview(event, tabname, filename) {
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea');
    var button = gradioApp().getElementById(tabname + '_save_preview');

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksSearchButton(tabs_id, event) {
    var searchTextarea = gradioApp().querySelector("#" + tabs_id + ' > label > textarea');
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
}

var globalPopup = null;
var globalPopupInner = null;

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}

function popup(contents) {
    if (!globalPopup) {
        globalPopup = document.createElement('div');
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement('div');
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner);

        gradioApp().querySelector('.main').appendChild(globalPopup);
    }

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);

    globalPopup.style.display = "flex";
}

var storedPopupIds = {};
function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    popup(storedPopupIds[id]);
}

function extraNetworksShowMetadata(text) {
    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
}

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksRequestMetadata(event, extraPage, cardName) {
    var showError = function() {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    requestGet("./sd_extra_networks/metadata", {page: extraPage, item: cardName}, function(data) {
        if (data && data.metadata) {
            extraNetworksShowMetadata(data.metadata);
        } else {
            showError();
        }
    }, showError);

    event.stopPropagation();
}

var extraPageUserMetadataEditors = {};

function extraNetworksEditUserMetadata(event, tabname, extraPage, cardName) {
    var id = tabname + '_' + extraPage + '_edit_user_metadata';

    var editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + ' textarea');
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }

    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);

    event.stopPropagation();
}

function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet("./sd_extra_networks/get-single-card", {page: page, tabname: tabname, name: name}, function(data) {
        if (data && data.html) {
            var card = gradioApp().querySelector(`#${tabname}_${page.replace(" ", "_")}_cards > .card[data-name="${name}"]`);

            var newDiv = document.createElement('DIV');
            newDiv.innerHTML = data.html;
            var newCard = newDiv.firstElementChild;

            newCard.style.display = '';
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        }
    });
}

window.addEventListener("keydown", function(event) {
    if (event.key == "Escape") {
        closePopup();
    }
});
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
