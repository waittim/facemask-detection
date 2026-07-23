# WearMask

> A privacy-first, real-time face mask detection system that runs entirely
> in the web browser using YOLO-Fastest, NCNN, C++, and WebAssembly.

[Live Demo](https://facemask-detection.com/) |
[Research Paper](https://doi.org/10.2352/EI.2023.35.11.HPCI-229) |
[Training Repository](https://github.com/waittim/mask-detector) |
[Dataset](https://github.com/waittim/mask-detector/tree/master/modeling/data) |
[How to Cite](#citation)

**`facemask-detection`** is the official WearMask project homepage and browser
deployment repository. **`mask-detector`** contains model training, data
preparation, evaluation, and conversion workflows.

## Project at a glance

- **Task:** Face mask object detection
- **Runtime:** Fully client-side web browser inference
- **Model:** YOLO-Fastest
- **Inference engine:** NCNN
- **Deployment target:** WebAssembly
- **Input:** Webcam video frames
- **Output:** Face bounding boxes and mask classifications (Mask / No Mask)
- **Server upload:** None
- **Languages:** English, Chinese, Spanish, French, German, Japanese, Portuguese, Korean, Italian, Russian
- **Research paper:** [WearMask (2023)](https://doi.org/10.2352/EI.2023.35.11.HPCI-229)

## What WearMask does

WearMask detects whether people in a webcam feed appear to be wearing a face
mask. Inference runs entirely in the user's browser. Camera frames are not
uploaded to an application server.

WearMask is a **research and demonstration system**, not a medical device or
diagnostic tool.

## Why it runs in the browser

- **Privacy:** Video stays on the user's device.
- **Serverless edge computing:** No GPU server or backend inference required.
- **Accessibility:** Works from a URL without installation.
- **Low deployment cost:** Static hosting via GitHub Pages.

## Features

- Real-time face mask detection via WebAssembly
- Real-time telemetry dashboard (FPS, inference latency, detection counts)
- 10-language UI via shared i18n module
- Fully client-side processing (privacy-first)
- Responsive design for desktop and mobile

## Architecture

```
Camera frame
  → browser canvas
  → C++ preprocessing
  → WebAssembly runtime
  → NCNN inference
  → YOLO-Fastest decoding
  → bounding boxes and classes
  → canvas rendering
```

Training and conversion pipeline (in `mask-detector`):

```
PyTorch training → Darknet weights → NCNN → optimized NCNN → WebAssembly
```

## Repository relationship

WearMask is maintained across two repositories:

| Repository | Role |
|------------|------|
| [`waittim/facemask-detection`](https://github.com/waittim/facemask-detection) | Production browser application, C++ inference code, NCNN assets, WebAssembly build, localization, and GitHub Pages deployment |
| [`waittim/mask-detector`](https://github.com/waittim/mask-detector) | Model training, dataset preparation, evaluation, model weights, and conversion steps |

The website and both repositories describe the same WearMask research system.

## Terminology

- **WearMask:** The complete research system and public web application.
- **Browser deployment:** The code in this repository.
- **Training pipeline:** The code in `waittim/mask-detector`.
- **Client-side inference:** Inference performed on the user's device without uploading camera frames to an application server.

## Model and training data

- Model architecture: YOLO-Fastest
- Training framework: PyTorch (see `mask-detector`)
- Training data: [`mask-detector/modeling/data`](https://github.com/waittim/mask-detector/tree/master/modeling/data)
- Deployed weights: `docs/static/model/` (pre-built NCNN assets compiled to WebAssembly)

## Privacy model

- All detection runs locally via WebAssembly.
- No camera images or video are uploaded to a server.
- No mask-wearing results are stored server-side.
- See [Privacy statement](https://facemask-detection.com/privacy/) for details.

## Known limitations

- WearMask is a research prototype and is not a medical device.
- Detection performance depends on camera quality, lighting, face size, pose, occlusion, and device performance.
- The system detects visual mask-wearing patterns; it does not determine whether a mask is medically effective or correctly fitted.
- Browser performance varies by CPU, browser, and WebAssembly support.

## Project Structure

```
facemask-detection/
├── docs/                 # Frontend files for GitHub Pages deployment
│   ├── index.html        # Main application page
│   ├── about/            # Project overview (static)
│   ├── research/         # Research and paper information
│   ├── citation/         # Citation instructions
│   ├── technical-overview/
│   ├── limitations/
│   ├── privacy/          # Privacy policy and legal service statement
│   ├── static/           # CSS, JS, images, WebAssembly model files
│   │   ├── js/           # site-common.js, i18n.js
│   │   └── model/        # yolo.wasm, yolo.js, yolo.data
│   ├── i18n/             # Locale dictionaries (10 languages)
│   ├── llms.txt          # LLM-oriented project index
│   ├── project.json      # Machine-readable project facts
│   ├── sitemap.xml
│   └── robots.txt
├── src/                  # C++ source code
│   ├── yolo.cpp
│   ├── CMakeLists.txt
│   ├── ncnn/             # Pre-built NCNN WebAssembly libraries
│   └── ios/
├── CITATION.cff          # GitHub citation metadata
├── codemeta.json         # Research software metadata
└── scripts/
    └── server.py         # Local development server (recommended)
```

## Usage

### Live demo (production)

Visit https://facemask-detection.com — no installation required.

### Run locally (recommended for development)

```bash
python3 scripts/server.py
```

Then open http://127.0.0.1:8888 (falls back to port 8889 if 8888 is in use).

This server sets the COOP/COEP headers required for WebAssembly threading.
Use the latest Chrome for best performance.

### GitHub Pages (fork / maintainer setup)

To deploy your own fork, configure GitHub Pages to serve from the `docs/` directory.

### Manual static server (limited)

```bash
cd docs
python3 -m http.server 8888
```

This works for basic static preview but does **not** set COOP/COEP headers.
WebAssembly features may fail; prefer `scripts/server.py` instead.

## Building from Source

Pre-built WebAssembly artifacts are already in `docs/static/model/`. Most users
do not need to rebuild.

To rebuild from C++ source:

1. Install the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
2. Build in `src/` (NCNN WebAssembly libraries are bundled under `src/ncnn/`):

```bash
cd src
emcmake cmake .
emmake make
```

3. Copy the generated `yolo.wasm`, `yolo.js`, and `yolo.data` to `docs/static/model/`

## Research paper

**Title:** WearMask: Fast in-browser face mask detection with serverless edge computing for COVID-19

**Authors:** Zekun Wang, Pengwei Wang, Peter C. Louis, Lee E. Wheless, Yuankai Huo

**Journal:** Electronic Imaging, Volume 35, Issue 11, 2023

**DOI:** https://doi.org/10.2352/EI.2023.35.11.HPCI-229

More details: https://facemask-detection.com/research/

## Citation

If you use WearMask in academic work, please cite the paper:

```bibtex
@article{
    wang_wang_louis_wheless_huo_2023,
    title={WearMask: Fast in-browser face mask detection with serverless edge computing for covid-19},
    volume={35},
    DOI={10.2352/ei.2023.35.11.hpci-229},
    number={11},
    journal={Electronic Imaging},
    author={Wang, Zekun and Wang, Pengwei and Louis, Peter C. and Wheless, Lee E. and Huo, Yuankai},
    year={2023}
}
```

When reusing implementation code, also link to the relevant repository:

- Deployment: https://github.com/waittim/facemask-detection
- Training: https://github.com/waittim/mask-detector

See also [`CITATION.cff`](CITATION.cff) and https://facemask-detection.com/citation/

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Maintainer

Zekun Wang — [GitHub](https://github.com/waittim) | [ORCID](https://orcid.org/0000-0001-8631-7022) | [LinkedIn](https://www.linkedin.com/in/zekun-wang/)

Vanderbilt University Data Science Institute

## For maintainers

Suggested GitHub repository **About** settings:

- **Description:** Privacy-first real-time face mask detection in the browser using YOLO-Fastest, NCNN, C++, and WebAssembly.
- **Website:** https://facemask-detection.com/
- **Topics:** `face-mask-detection`, `computer-vision`, `object-detection`, `yolo`, `yolo-fastest`, `ncnn`, `webassembly`, `wasm`, `browser-ml`, `client-side-ai`, `privacy-preserving-ai`, `edge-ai`, `serverless-inference`, `research-software`, `covid-19`
