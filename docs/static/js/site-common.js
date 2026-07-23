/* WearMask Site-Wide Shared Interactions (Theme & i18n Dropdown & Subpage Translation) */
(function() {
    var THEME_MODES = ['auto', 'light', 'dark'];
    var LANG_DISPLAY_NAMES = {
        'en': 'English',
        'es': 'Español',
        'zh': '中文'
    };

    function initSiteControls() {
        // Theme Cycling Logic
        var btnThemeCycle = document.getElementById('btn-theme-cycle');
        var iconAuto = document.getElementById('theme-icon-auto');
        var iconLight = document.getElementById('theme-icon-light');
        var iconDark = document.getElementById('theme-icon-dark');

        function applyThemeMode(mode) {
            var isDark = false;
            if (mode === 'auto') {
                isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            } else {
                isDark = mode === 'dark';
            }

            var htmlEl = document.documentElement;
            if (isDark) {
                htmlEl.classList.remove('light');
                htmlEl.classList.add('dark');
            } else {
                htmlEl.classList.remove('dark');
                htmlEl.classList.add('light');
            }

            localStorage.setItem('wearmask.theme', mode);
            if (iconAuto) iconAuto.classList.toggle('hidden', mode !== 'auto');
            if (iconLight) iconLight.classList.toggle('hidden', mode !== 'light');
            if (iconDark) iconDark.classList.toggle('hidden', mode !== 'dark');

            if (btnThemeCycle) {
                if (mode === 'auto') btnThemeCycle.title = 'System Theme (Click to switch)';
                else if (mode === 'light') btnThemeCycle.title = 'Light Mode (Click to switch)';
                else if (mode === 'dark') btnThemeCycle.title = 'Dark Mode (Click to switch)';
            }
        }

        var currentMode = localStorage.getItem('wearmask.theme') || 'auto';
        applyThemeMode(currentMode);

        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
                if ((localStorage.getItem('wearmask.theme') || 'auto') === 'auto') {
                    applyThemeMode('auto');
                }
            });
        }

        if (btnThemeCycle) {
            btnThemeCycle.addEventListener('click', function() {
                var current = localStorage.getItem('wearmask.theme') || 'auto';
                var idx = THEME_MODES.indexOf(current);
                var nextMode = THEME_MODES[(idx + 1) % THEME_MODES.length];
                applyThemeMode(nextMode);
            });
        }

        // Detect Language
        var params = new URLSearchParams(window.location.search);
        var urlLang = params.get('lang');
        var currentLang = 'en';

        if (urlLang && ['en', 'es', 'zh'].indexOf(urlLang) !== -1) {
            currentLang = urlLang;
            localStorage.setItem('wearmask.lang', currentLang);
        } else {
            var storedLang = localStorage.getItem('wearmask.lang');
            if (storedLang && ['en', 'es', 'zh'].indexOf(storedLang) !== -1) {
                currentLang = storedLang;
            } else {
                var browserLang = (navigator.language || navigator.userLanguage || '').toLowerCase();
                if (browserLang.indexOf('zh') === 0) currentLang = 'zh';
                else if (browserLang.indexOf('es') === 0) currentLang = 'es';
                else currentLang = 'en';
            }
        }

        // Language Dropdown Logic
        var dropdownBtn = document.getElementById('lang-dropdown-btn');
        var dropdownMenu = document.getElementById('lang-dropdown-menu');
        var optionBtns = document.querySelectorAll('.lang-option-btn');
        var currentLabel = document.getElementById('lang-current-label');

        if (currentLabel) currentLabel.textContent = LANG_DISPLAY_NAMES[currentLang] || 'English';

        optionBtns.forEach(function(btn) {
            var isActive = btn.getAttribute('data-lang') === currentLang;
            var check = btn.querySelector('.lang-check');
            if (check) check.classList.toggle('hidden', !isActive);
            btn.classList.toggle('font-bold', isActive);
            btn.classList.toggle('text-cyan-400', isActive);

            btn.addEventListener('click', function() {
                var selectedLang = btn.getAttribute('data-lang');
                if (selectedLang) {
                    localStorage.setItem('wearmask.lang', selectedLang);
                    var url = new URL(window.location.href);
                    url.searchParams.set('lang', selectedLang);
                    window.location.href = url.toString();
                }
            });
        });

        if (dropdownBtn && dropdownMenu) {
            dropdownBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                var isOpen = !dropdownMenu.classList.contains('hidden');
                dropdownMenu.classList.toggle('hidden', isOpen);
                dropdownBtn.setAttribute('aria-expanded', !isOpen);
            });

            document.addEventListener('click', function(e) {
                if (!dropdownMenu.classList.contains('hidden') && !dropdownBtn.contains(e.target) && !dropdownMenu.contains(e.target)) {
                    dropdownMenu.classList.add('hidden');
                    dropdownBtn.setAttribute('aria-expanded', 'false');
                }
            });
        }

        // Dynamic i18n Translation for Subpages
        var isSubpage = document.querySelector('script[src*="../static/js/site-common.js"]') !== null;
        var fetchPath = isSubpage ? '../i18n/' + currentLang + '.json' : 'i18n/' + currentLang + '.json';

        fetch(fetchPath)
            .then(function(res) { return res.json(); })
            .then(function(dict) {
                if (!dict) return;
                applyDictionary(dict);
            })
            .catch(function(err) {
                console.warn('Failed to load subpage i18n dictionary:', err);
            });

        function getNestedValue(obj, path) {
            return path.split('.').reduce(function(acc, key) {
                return acc && acc[key] !== undefined ? acc[key] : undefined;
            }, obj);
        }

        function applyDictionary(data) {
            document.querySelectorAll('[data-i18n]').forEach(function(el) {
                var key = el.getAttribute('data-i18n');
                var val = getNestedValue(data, key);
                if (val !== undefined && val !== null && typeof val === 'string') {
                    el.textContent = val;
                }
            });

            document.querySelectorAll('[data-i18n-html]').forEach(function(el) {
                var key = el.getAttribute('data-i18n-html');
                var val = getNestedValue(data, key);
                if (val !== undefined && val !== null && typeof val === 'string') {
                    el.innerHTML = val;
                }
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSiteControls);
    } else {
        initSiteControls();
    }
})();
