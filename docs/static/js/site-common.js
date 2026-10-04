/* WearMask site-wide shared interactions: theme cycle + i18n bootstrap */
(function() {
    'use strict';

    // Multimodal Feedback: Apple-style subtle, non-intrusive haptic cues
    window.WearMaskHaptics = {
        trigger: function(type) {
            if (typeof navigator !== 'undefined' && navigator.vibrate) {
                try {
                    if (type === 'light') navigator.vibrate(8);
                    else if (type === 'medium') navigator.vibrate(14);
                    else if (type === 'heavy') navigator.vibrate(24);
                    else if (type === 'selection') navigator.vibrate(6);
                } catch (e) {}
            }
        }
    };

    var THEME_MODES = ['auto', 'light', 'dark'];

    function initTheme() {
        var btnThemeCycle = document.getElementById('btn-theme-cycle');
        var iconAuto = document.getElementById('theme-icon-auto');
        var iconLight = document.getElementById('theme-icon-light');
        var iconDark = document.getElementById('theme-icon-dark');

        function themeTitle(mode) {
            if (window.WearMaskI18n) {
                if (mode === 'auto') return WearMaskI18n.t('toast.themeAuto', 'Theme: System');
                if (mode === 'light') return WearMaskI18n.t('toast.themeLight', 'Theme: Light');
                if (mode === 'dark') return WearMaskI18n.t('toast.themeDark', 'Theme: Dark');
            }
            if (mode === 'auto') return 'Theme: System';
            if (mode === 'light') return 'Theme: Light';
            return 'Theme: Dark';
        }

        function applyThemeMode(mode) {
            var isDark = false;
            if (mode === 'auto') {
                isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            } else {
                isDark = mode === 'dark';
            }

            var htmlEl = document.documentElement;
            var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            if (reduceMotion) {
                htmlEl.style.transition = 'none';
            }

            if (isDark) {
                htmlEl.classList.remove('light');
                htmlEl.classList.add('dark');
            } else {
                htmlEl.classList.remove('dark');
                htmlEl.classList.add('light');
            }

            try { localStorage.setItem('wearmask.theme', mode); } catch (e) {}
            if (iconAuto) iconAuto.classList.toggle('hidden', mode !== 'auto');
            if (iconLight) iconLight.classList.toggle('hidden', mode !== 'light');
            if (iconDark) iconDark.classList.toggle('hidden', mode !== 'dark');

            if (btnThemeCycle) {
                var title = themeTitle(mode);
                btnThemeCycle.title = title;
                btnThemeCycle.setAttribute('aria-label', title);
                btnThemeCycle.setAttribute('data-theme-mode', mode);
            }

            if (reduceMotion) {
                requestAnimationFrame(function() {
                    htmlEl.style.transition = '';
                });
            }
        }

        var currentMode = 'auto';
        try { currentMode = localStorage.getItem('wearmask.theme') || 'auto'; } catch (e) {}
        applyThemeMode(currentMode);

        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
                var mode = 'auto';
                try { mode = localStorage.getItem('wearmask.theme') || 'auto'; } catch (e) {}
                if (mode === 'auto') applyThemeMode('auto');
            });
        }

        if (btnThemeCycle) {
            btnThemeCycle.addEventListener('click', function() {
                window.WearMaskHaptics.trigger('selection');
                var current = 'auto';
                try { current = localStorage.getItem('wearmask.theme') || 'auto'; } catch (e) {}
                var idx = THEME_MODES.indexOf(current);
                var nextMode = THEME_MODES[(idx + 1) % THEME_MODES.length];
                applyThemeMode(nextMode);
            });
        }
    }

    function resolveI18nBasePath() {
        var scripts = document.getElementsByTagName('script');
        for (var i = 0; i < scripts.length; i++) {
            var src = scripts[i].getAttribute('src') || '';
            if (src.indexOf('site-common.js') !== -1 || src.indexOf('i18n.js') !== -1) {
                if (src.indexOf('../') !== -1) return '../i18n/';
                return 'i18n/';
            }
        }
        var path = window.location.pathname || '';
        if (/\/(about|privacy|technical-overview|research|citation|limitations)(\/|$)/.test(path)) {
            return '../i18n/';
        }
        return 'i18n/';
    }

    function initChromeFlagsCopy() {
        function copyTextToClipboard(text) {
            if (navigator.clipboard && window.isSecureContext) {
                return navigator.clipboard.writeText(text);
            }
            return new Promise(function(resolve, reject) {
                var textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.setAttribute('readonly', '');
                textarea.style.position = 'fixed';
                textarea.style.top = '-9999px';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    var ok = document.execCommand('copy');
                    document.body.removeChild(textarea);
                    if (ok) resolve();
                    else reject(new Error('Copy command failed'));
                } catch (err) {
                    document.body.removeChild(textarea);
                    reject(err);
                }
            });
        }

        function syncTips() {
            var copyLabel = window.WearMaskI18n ? WearMaskI18n.t('setup.copyFlags', 'Copy') : 'Copy';
            var copiedLabel = window.WearMaskI18n ? WearMaskI18n.t('setup.copiedFlags', 'Copied!') : 'Copied!';
            document.querySelectorAll('.chrome-flags-code').forEach(function(el) {
                el.setAttribute('data-tip', copyLabel);
                el.setAttribute('data-tip-copied', copiedLabel);
                el.setAttribute('aria-label', copyLabel + ' chrome://flags');
            });
        }

        function copyChromeFlags(el) {
            if (el.classList.contains('is-copied')) return;
            copyTextToClipboard('chrome://flags').then(function() {
                window.WearMaskHaptics.trigger('medium');
                el.classList.add('is-copied');
                setTimeout(function() {
                    el.classList.remove('is-copied');
                }, 2000);
            }).catch(function(err) {
                console.error('Failed to copy chrome://flags', err);
            });
        }

        document.addEventListener('click', function(e) {
            var target = e.target.closest && e.target.closest('.chrome-flags-code');
            if (!target) return;
            e.preventDefault();
            copyChromeFlags(target);
        });

        document.addEventListener('keydown', function(e) {
            var target = e.target.closest && e.target.closest('.chrome-flags-code');
            if (!target) return;
            if (e.key !== 'Enter' && e.key !== ' ') return;
            e.preventDefault();
            copyChromeFlags(target);
        });

        return syncTips;
    }

    // Smooth Details Accordion Disclosure (Progressive Enhancement)
    function initSmoothDetails() {
        var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduceMotion) return;
        // If the browser natively supports ::details-content interpolation, let CSS handle it
        if (window.CSS && CSS.supports && (CSS.supports('interpolate-size: allow-keywords') || CSS.supports('selector(::details-content)'))) {
            return;
        }

        var detailsElements = document.querySelectorAll('details.feature-card, details.accordion-card');
        detailsElements.forEach(function(detail) {
            var summary = detail.querySelector('summary');
            if (!summary) return;

            var isClosing = false;
            var isExpanding = false;
            var animation = null;

            summary.addEventListener('click', function(e) {
                e.preventDefault();
                detail.style.overflow = 'hidden';

                if (isClosing || !detail.open) {
                    openDetail();
                } else if (isExpanding || detail.open) {
                    closeDetail();
                }
            });

            function closeDetail() {
                isClosing = true;
                window.WearMaskHaptics.trigger('selection');
                var startHeight = detail.offsetHeight + 'px';
                var endHeight = summary.offsetHeight + 'px';

                if (animation) animation.cancel();
                animation = detail.animate({
                    height: [startHeight, endHeight]
                }, {
                    duration: 280,
                    easing: 'cubic-bezier(0.32, 0.72, 0, 1)'
                });

                animation.onfinish = function() {
                    detail.open = false;
                    animation = null;
                    isClosing = false;
                    detail.style.height = '';
                    detail.style.overflow = '';
                };
                animation.oncancel = function() {
                    isClosing = false;
                };
            }

            function openDetail() {
                isExpanding = true;
                window.WearMaskHaptics.trigger('selection');
                detail.style.height = detail.offsetHeight + 'px';
                detail.open = true;

                window.requestAnimationFrame(function() {
                    var startHeight = detail.offsetHeight + 'px';
                    var endHeight = detail.scrollHeight + 'px';

                    if (animation) animation.cancel();
                    animation = detail.animate({
                        height: [startHeight, endHeight]
                    }, {
                        duration: 280,
                        easing: 'cubic-bezier(0.32, 0.72, 0, 1)'
                    });

                    animation.onfinish = function() {
                        animation = null;
                        isExpanding = false;
                        detail.style.height = '';
                        detail.style.overflow = '';
                    };
                    animation.oncancel = function() {
                        isExpanding = false;
                    };
                });
            }
        });
    }

    function init() {
        initTheme();
        var syncChromeTips = initChromeFlagsCopy();
        initSmoothDetails();
        if (window.WearMaskI18n) {
            WearMaskI18n.init({
                basePath: resolveI18nBasePath(),
                onReady: function(dict, lang) {
                    syncChromeTips();
                    if (typeof window.WearMaskOnI18nReady === 'function') {
                        window.WearMaskOnI18nReady(dict, lang);
                    }
                },
                onLangChange: function(dict, lang) {
                    syncChromeTips();
                    if (typeof window.WearMaskOnI18nChange === 'function') {
                        window.WearMaskOnI18nChange(dict, lang);
                    }
                }
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
