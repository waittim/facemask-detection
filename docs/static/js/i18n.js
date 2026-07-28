/**
 * WearMask shared i18n: locale registry, detection, dictionary apply, hreflang, lang UI.
 */
(function(global) {
    'use strict';

    var VERSION = '3.3.0';
    var STORAGE_KEY = 'wearmask.lang';
    var DEFAULT_LANG = 'en';
    var SITE_ORIGIN = 'https://facemask-detection.com';

    var LOCALES = {
        en: { bcp47: 'en', nativeName: 'English', dir: 'ltr' },
        zh: { bcp47: 'zh-Hans', nativeName: '中文', dir: 'ltr' },
        es: { bcp47: 'es', nativeName: 'Español', dir: 'ltr' },
        fr: { bcp47: 'fr', nativeName: 'Français', dir: 'ltr' },
        de: { bcp47: 'de', nativeName: 'Deutsch', dir: 'ltr' },
        ja: { bcp47: 'ja', nativeName: '日本語', dir: 'ltr' },
        pt: { bcp47: 'pt', nativeName: 'Português', dir: 'ltr' },
        ko: { bcp47: 'ko', nativeName: '한국어', dir: 'ltr' },
        it: { bcp47: 'it', nativeName: 'Italiano', dir: 'ltr' },
        ru: { bcp47: 'ru', nativeName: 'Русский', dir: 'ltr' }
    };

    var SUPPORTED = Object.keys(LOCALES);
    var state = {
        lang: DEFAULT_LANG,
        dict: null,
        fallbackDict: null,
        basePath: 'i18n/',
        ready: null,
        onLangChange: null
    };

    function normalizeLang(lang) {
        if (!lang) return null;
        lang = String(lang).toLowerCase().replace('_', '-');
        if (SUPPORTED.indexOf(lang) !== -1) return lang;
        var primary = lang.split('-')[0];
        if (primary === 'zh') return 'zh';
        if (SUPPORTED.indexOf(primary) !== -1) return primary;
        return null;
    }

    function detectLanguage() {
        var params = new URLSearchParams(global.location.search);
        var urlLang = normalizeLang(params.get('lang'));
        if (urlLang) {
            try { localStorage.setItem(STORAGE_KEY, urlLang); } catch (e) {}
            return urlLang;
        }
        try {
            var stored = normalizeLang(localStorage.getItem(STORAGE_KEY));
            if (stored) return stored;
        } catch (e) {}
        var nav = normalizeLang(navigator.language || navigator.userLanguage);
        if (nav) return nav;
        if (navigator.languages) {
            for (var i = 0; i < navigator.languages.length; i++) {
                var candidate = normalizeLang(navigator.languages[i]);
                if (candidate) return candidate;
            }
        }
        return DEFAULT_LANG;
    }

    function applyDocumentLang(lang) {
        var meta = LOCALES[lang] || LOCALES[DEFAULT_LANG];
        var html = document.documentElement;
        html.lang = meta.bcp47;
        html.dir = meta.dir || 'ltr';
        html.setAttribute('data-lang', lang);
    }

    function getValue(obj, path) {
        if (!obj || !path) return undefined;
        return path.split('.').reduce(function(acc, key) {
            return acc && acc[key] !== undefined ? acc[key] : undefined;
        }, obj);
    }

    function t(path, fallback) {
        var value = getValue(state.dict, path);
        if (value === undefined || value === null) {
            value = getValue(state.fallbackDict, path);
        }
        if (value === undefined || value === null) return fallback;
        return value;
    }

    function applyTranslations(dict, fallbackDict) {
        state.dict = dict || {};
        state.fallbackDict = fallbackDict || state.fallbackDict || {};

        if (dict) {
            var titleKey = document.documentElement.getAttribute('data-i18n-title-key');
            var descKey = document.documentElement.getAttribute('data-i18n-desc-key');
            var title = titleKey ? t(titleKey) : (dict.page && dict.page.title);
            var description = descKey ? t(descKey) : null;
            // Homepage (no page-specific keys) may also update the default meta description.
            if (!titleKey && !descKey && dict.page && dict.page.description) {
                description = dict.page.description;
            }
            if (title && typeof title === 'string') document.title = title;
            if (description && typeof description === 'string') {
                var meta = document.querySelector('meta[name="description"]');
                if (meta) meta.setAttribute('content', description);
            }
        }

        document.querySelectorAll('[data-i18n]').forEach(function(el) {
            var value = t(el.getAttribute('data-i18n'));
            if (value !== undefined && value !== null && typeof value !== 'object') {
                el.textContent = value;
            }
        });

        document.querySelectorAll('[data-i18n-html]').forEach(function(el) {
            var value = t(el.getAttribute('data-i18n-html'));
            if (typeof value === 'string') {
                el.innerHTML = value;
            }
        });

        document.querySelectorAll('[data-i18n-attr]').forEach(function(el) {
            var mappings = el.getAttribute('data-i18n-attr').split('|');
            mappings.forEach(function(mapping) {
                var parts = mapping.split(':');
                if (parts.length !== 2) return;
                var attr = parts[0].trim();
                var key = parts[1].trim();
                var value = t(key);
                if (attr && value !== undefined && value !== null && typeof value !== 'object') {
                    el.setAttribute(attr, value);
                }
            });
        });
    }

    function fetchJson(url) {
        return fetch(url).then(function(res) {
            if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
            return res.json();
        });
    }

    function loadTranslations(lang) {
        var primaryUrl = state.basePath + lang + '.json?v=' + VERSION;
        var fallbackUrl = state.basePath + DEFAULT_LANG + '.json?v=' + VERSION;

        var primaryPromise = fetchJson(primaryUrl).catch(function() { return null; });
        var fallbackPromise = lang === DEFAULT_LANG
            ? primaryPromise
            : fetchJson(fallbackUrl).catch(function() { return {}; });

        return Promise.all([primaryPromise, fallbackPromise]).then(function(pair) {
            var primary = pair[0];
            var fallback = pair[1] || {};
            if (!primary) {
                primary = fallback;
                lang = DEFAULT_LANG;
                state.lang = lang;
                applyDocumentLang(lang);
            }
            state.fallbackDict = fallback;
            applyTranslations(primary, fallback);
            return primary || {};
        });
    }

    function withLangParam(href, lang) {
        try {
            var url = new URL(href, global.location.href);
            if (url.origin !== global.location.origin && url.origin !== SITE_ORIGIN && !url.protocol.startsWith('http')) {
                return href;
            }
            if (url.origin !== global.location.origin && url.hostname !== 'facemask-detection.com' && url.hostname !== global.location.hostname) {
                return href;
            }
            url.searchParams.set('lang', lang);
            return url.pathname + url.search + url.hash;
        } catch (e) {
            return href;
        }
    }

    function rewriteInternalLinks(lang) {
        document.querySelectorAll('a[href]').forEach(function(anchor) {
            var href = anchor.getAttribute('href');
            if (!href || href.charAt(0) === '#' || href.indexOf('mailto:') === 0 || href.indexOf('javascript:') === 0) {
                return;
            }
            if (anchor.hasAttribute('data-i18n-skip-lang')) return;
            try {
                var url = new URL(href, global.location.href);
                var sameHost = url.hostname === global.location.hostname || url.hostname === 'facemask-detection.com';
                if (!sameHost) return;
                if (/\.(png|jpe?g|gif|webp|svg|pdf|json|wasm|js|css)$/i.test(url.pathname)) return;
                anchor.setAttribute('href', withLangParam(href, lang));
            } catch (e) {}
        });
    }

    function injectHreflang() {
        var head = document.head;
        if (!head) return;
        head.querySelectorAll('link[data-wearmask-hreflang]').forEach(function(el) { el.remove(); });

        var path = global.location.pathname || '/';
        var canonicalBase = SITE_ORIGIN + path;

        SUPPORTED.forEach(function(code) {
            var link = document.createElement('link');
            link.rel = 'alternate';
            link.hreflang = LOCALES[code].bcp47;
            link.href = canonicalBase + '?lang=' + code;
            link.setAttribute('data-wearmask-hreflang', code);
            head.appendChild(link);
        });

        var xDefault = document.createElement('link');
        xDefault.rel = 'alternate';
        xDefault.hreflang = 'x-default';
        xDefault.href = canonicalBase;
        xDefault.setAttribute('data-wearmask-hreflang', 'x-default');
        head.appendChild(xDefault);
    }

    function buildLangMenu(lang) {
        var menu = document.getElementById('lang-dropdown-menu');
        if (!menu) return;

        menu.innerHTML = '';
        menu.classList.add('lang-dropdown-scroll');
        SUPPORTED.forEach(function(code) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'lang-option-btn w-full text-left px-3.5 py-2 text-xs text-gray-300 hover:text-white hover:bg-white/10 flex items-center justify-between transition';
            btn.setAttribute('data-lang', code);
            btn.setAttribute('role', 'option');
            btn.setAttribute('aria-selected', code === lang ? 'true' : 'false');
            btn.setAttribute('tabindex', '-1');

            var label = document.createElement('span');
            label.textContent = LOCALES[code].nativeName;
            btn.appendChild(label);

            var check = document.createElement('span');
            check.className = 'lang-check text-cyan-400' + (code === lang ? '' : ' hidden');
            check.setAttribute('aria-hidden', 'true');
            check.textContent = '\u2713';
            btn.appendChild(check);

            if (code === lang) {
                btn.classList.add('font-bold', 'text-cyan-400');
            }

            btn.addEventListener('click', function() {
                switchLanguage(code);
            });
            menu.appendChild(btn);
        });
    }

    function updateLangLabel(lang) {
        var labelEl = document.getElementById('lang-current-label');
        if (labelEl) {
            labelEl.textContent = (LOCALES[lang] && LOCALES[lang].nativeName) || 'English';
        }
        var btn = document.getElementById('lang-dropdown-btn');
        if (btn) {
            btn.setAttribute('title', 'Change language');
            btn.setAttribute('aria-label', 'Change language');
        }
    }

    function closeLangMenu(dropdownMenu, dropdownBtn, returnFocus) {
        if (!dropdownMenu) return;
        dropdownMenu.classList.remove('is-open');
        if (dropdownBtn) dropdownBtn.setAttribute('aria-expanded', 'false');
        if (returnFocus && dropdownBtn) dropdownBtn.focus();
    }

    function openLangMenu(dropdownMenu, dropdownBtn, focusSelected) {
        if (!dropdownMenu || !dropdownBtn) return;
        dropdownMenu.classList.remove('hidden');
        // Force reflow so enter transition runs from the closed presentation value
        void dropdownMenu.offsetWidth;
        dropdownMenu.classList.add('is-open');
        dropdownBtn.setAttribute('aria-expanded', 'true');
        var options = Array.prototype.slice.call(dropdownMenu.querySelectorAll('[role="option"]'));
        if (!options.length) return;
        var selected = dropdownMenu.querySelector('[aria-selected="true"]') || options[0];
        if (focusSelected) selected.focus();
    }

    function initLangDropdown(lang) {
        buildLangMenu(lang);
        updateLangLabel(lang);

        var dropdownBtn = document.getElementById('lang-dropdown-btn');
        var dropdownMenu = document.getElementById('lang-dropdown-menu');
        if (!dropdownBtn || !dropdownMenu) return;
        if (dropdownMenu.getAttribute('data-lang-ui-bound') === '1') return;
        dropdownMenu.setAttribute('data-lang-ui-bound', '1');
        dropdownMenu.classList.add('lang-dropdown-panel');
        // Prefer is-open animation over display:none utility
        dropdownMenu.classList.remove('hidden');

        function optionButtons() {
            return Array.prototype.slice.call(dropdownMenu.querySelectorAll('[role="option"]'));
        }

        function moveFocus(delta) {
            var options = optionButtons();
            if (!options.length) return;
            var idx = options.indexOf(document.activeElement);
            if (idx === -1) idx = options.findIndex(function(el) { return el.getAttribute('aria-selected') === 'true'; });
            if (idx === -1) idx = 0;
            var next = (idx + delta + options.length) % options.length;
            options[next].focus();
        }

        dropdownBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            var isOpen = dropdownMenu.classList.contains('is-open');
            if (isOpen) closeLangMenu(dropdownMenu, dropdownBtn, false);
            else openLangMenu(dropdownMenu, dropdownBtn, true);
        });

        dropdownBtn.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openLangMenu(dropdownMenu, dropdownBtn, true);
            } else if (e.key === 'Escape') {
                closeLangMenu(dropdownMenu, dropdownBtn, false);
            }
        });

        dropdownMenu.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                e.preventDefault();
                closeLangMenu(dropdownMenu, dropdownBtn, true);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                moveFocus(1);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                moveFocus(-1);
            } else if (e.key === 'Home') {
                e.preventDefault();
                var first = optionButtons()[0];
                if (first) first.focus();
            } else if (e.key === 'End') {
                e.preventDefault();
                var opts = optionButtons();
                if (opts.length) opts[opts.length - 1].focus();
            } else if (e.key === 'Tab') {
                closeLangMenu(dropdownMenu, dropdownBtn, false);
            }
        });

        document.addEventListener('click', function(e) {
            if (dropdownMenu.classList.contains('is-open') &&
                !dropdownBtn.contains(e.target) &&
                !dropdownMenu.contains(e.target)) {
                closeLangMenu(dropdownMenu, dropdownBtn, false);
            }
        });
    }

    function switchLanguage(lang) {
        lang = normalizeLang(lang);
        if (!lang) return;

        var menu = document.getElementById('lang-dropdown-menu');
        var btn = document.getElementById('lang-dropdown-btn');
        closeLangMenu(menu, btn, false);

        if (lang === state.lang) return;

        try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}

        state.lang = lang;
        applyDocumentLang(lang);

        try {
            var url = new URL(global.location.href);
            url.searchParams.set('lang', lang);
            if (global.history && global.history.replaceState) {
                global.history.replaceState(null, '', url.pathname + url.search + url.hash);
            }
        } catch (e) {}

        loadTranslations(lang).then(function(dict) {
            buildLangMenu(lang);
            updateLangLabel(lang);
            rewriteInternalLinks(lang);
            injectHreflang();
            if (typeof state.onLangChange === 'function') {
                state.onLangChange(dict || {}, lang);
            }
        });
    }

    function revealPage() {
        document.documentElement.classList.remove('i18n-pending');
        if (document.body) document.body.classList.remove('i18n-loading');
    }

    function ready(fn) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fn);
        } else {
            fn();
        }
    }

    /**
     * @param {Object} options
     * @param {string} [options.basePath] Relative path to i18n/ folder
     * @param {function} [options.onReady] Called with dictionary after apply
     * @returns {Promise<Object>}
     */
    function init(options) {
        options = options || {};
        if (options.basePath) state.basePath = options.basePath;
        if (typeof options.onLangChange === 'function') {
            state.onLangChange = options.onLangChange;
        }

        state.lang = detectLanguage();
        applyDocumentLang(state.lang);
        injectHreflang();

        state.ready = loadTranslations(state.lang)
            .then(function(dict) {
                ready(function() {
                    initLangDropdown(state.lang);
                    rewriteInternalLinks(state.lang);
                    revealPage();
                    if (typeof options.onReady === 'function') {
                        options.onReady(dict || {}, state.lang);
                    }
                });
                return dict || {};
            })
            .catch(function(err) {
                console.error('WearMask i18n failed', err);
                ready(function() {
                    initLangDropdown(state.lang);
                    revealPage();
                    if (typeof options.onReady === 'function') {
                        options.onReady({}, state.lang);
                    }
                });
                return {};
            });

        return state.ready;
    }

    // Early boot helper (callable from inline <head> script)
    function bootEarly() {
        var lang = detectLanguage();
        applyDocumentLang(lang);
        document.documentElement.classList.add('i18n-pending');
        return lang;
    }

    global.WearMaskI18n = {
        VERSION: VERSION,
        STORAGE_KEY: STORAGE_KEY,
        DEFAULT_LANG: DEFAULT_LANG,
        LOCALES: LOCALES,
        SUPPORTED: SUPPORTED.slice(),
        normalizeLang: normalizeLang,
        detectLanguage: detectLanguage,
        applyDocumentLang: applyDocumentLang,
        getValue: getValue,
        t: t,
        applyTranslations: applyTranslations,
        loadTranslations: loadTranslations,
        switchLanguage: switchLanguage,
        init: init,
        bootEarly: bootEarly,
        getLang: function() { return state.lang; },
        getDict: function() { return state.dict; }
    };
})(typeof window !== 'undefined' ? window : this);
