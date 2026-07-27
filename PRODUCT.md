# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are people who want to try face-mask detection immediately in a browser (research, education, demonstration, or personal curiosity), without installing software or uploading video.

Secondary audiences include researchers and engineers evaluating client-side CV / WebAssembly deployment, and readers of the peer-reviewed WearMask paper who want a live companion demo.

## Product Purpose

WearMask detects whether faces in a live webcam feed appear to be wearing a mask, and labels results in real time (Mask / No Mask). Success means: open a URL, allow camera access, and see local inference with no application-server upload of frames.

WearMask is a research and demonstration system, not a medical device or diagnostic tool.

## Positioning

Inference runs entirely on-device in the browser via YOLO-Fastest + NCNN compiled to WebAssembly. Neighboring cloud vision products cannot truthfully claim the same "frames never leave the device for app inference" deployment for this demo.

## Operating Context

- Entry: static site (GitHub Pages / local static server)
- Input: webcam frames via `getUserMedia`
- Output: canvas overlays, status banner, optional telemetry (FPS, latency, counts)
- Companion repos: training / conversion in `mask-detector`; this repo hosts browser deployment and site
- Languages: en, zh, es, fr, de, ja, pt, ko, it, ru

## Capabilities and Constraints

**Capabilities**
- Real-time in-browser detection with pause / flip camera / snapshot / telemetry panel
- Smooth playback mode that decouples rendering from inference
- SIMD / multi-thread WASM path when available; iOS-oriented fallback build otherwise
- Theme modes: auto / light / dark
- Shared i18n across main and documentation pages

**Constraints**
- Requires JavaScript, WebAssembly, and camera permission
- Frame rate depends on device CPU; Safari/iOS may run slower
- No backend inference API in the product path
- Not for clinical diagnosis

**Open / undecided**
- Formal target WCAG level beyond current AA-oriented hardening (inferred goal: WCAG 2.2 AA for interactive chrome)

## Brand Commitments

- Product name: **WearMask**
- Domain / demo: https://facemask-detection.com/
- Logo: `docs/static/images/logo_mask_v2.svg`
- Voice: clear, technical, privacy-forward; research disclaimer required near privacy claims
- Institutional marks in footer: NSF, Vanderbilt (attribution, not product co-branding of the UI chrome)

## Evidence on Hand

- Live demo site and static docs under `docs/`
- Peer-reviewed paper: https://doi.org/10.2352/EI.2023.35.11.HPCI-229
- Source: https://github.com/waittim/facemask-detection
- Training repo: https://github.com/waittim/mask-detector
- Do not fabricate benchmarks, clinical outcomes, or customer testimonials

## Product Principles

1. **Local by default** — camera frames stay on-device for detection.
2. **Demo honesty** — research/demo status is visible; never imply medical certification.
3. **URL accessibility** — no install; works as a static web app.
4. **Progressive performance** — use SIMD/threads when present; degrade gracefully.
5. **Readable across locales** — UI strings go through the shared i18n system.

## Accessibility & Inclusion

Inferred product requirement: keyboard-operable controls, visible focus, status announcements for detection/toast changes, touch targets ≥44px, and contrast meeting WCAG AA for text and links in both themes. No confirmed specialized AT audience beyond general public web use.

---

*Note: This PRODUCT.md was authored from repository evidence (README, docs site, code) under an explicit request to run `$impeccable init` as part of best-practice remediation. Facts marked inferred can be corrected without a redesign.*
