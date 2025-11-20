# WearMask: In-browser Face Mask Detection

For more details about this project, please visit the [main repo](https://github.com/waittim/mask-detector).

## Project Structure

```
facemask-detection/
├── docs/                 # Frontend files for GitHub Pages deployment
│   ├── index.html       # Main application page
│   ├── static/          # Static assets
│   │   ├── css/         # Stylesheets
│   │   ├── js/          # JavaScript files
│   │   ├── images/      # Image assets
│   │   └── model/       # WebAssembly model files
│   ├── privacy/         # Privacy statement page
│   └── i18n/            # Internationalization files
├── src/                 # C++ source code
│   ├── yolo.cpp         # Main C++ implementation
│   ├── CMakeLists.txt   # Build configuration
│   ├── ncnn/            # ncnn library
│   └── ios/             # iOS-specific build files
└── scripts/             # Utility scripts
    └── server.py        # Local development server
```

## Usage

### Method 1: GitHub Pages (Production)

1. Configure GitHub Pages to serve from the `docs/` directory:
   - Go to repository Settings → Pages
   - Set Source to `/docs` directory
2. Visit https://facemask-detection.com (or your GitHub Pages URL)

### Method 2: Run Locally (Development)

1. Start the local development server:
   ```bash
   python3 scripts/server.py
   ```
   The server will automatically serve from the `docs/` directory.

2. Open your browser and navigate to:
   - http://localhost:8888

**Note:** For best performance, use Chrome with WebAssembly features enabled:
- Open `chrome://flags`
- Enable experimental WebAssembly features
- Restart Chrome

### Method 3: Manual Server

If you prefer to use Python's built-in server:

```bash
cd docs
python3 -m http.server 8888
```

Then open http://localhost:8888 in your browser.

## Building from Source

To rebuild the WebAssembly files from C++ source:

1. Install Emscripten SDK
2. Navigate to `src/` directory
3. Run CMake build:
   ```bash
   cd src
   emcmake cmake .
   emmake make
   ```
4. Copy generated `.wasm`, `.js`, and `.data` files to `docs/static/model/`

## Features

- Real-time face mask detection using WebAssembly
- Multi-language support (English, Spanish, Chinese)
- Fully client-side processing (privacy-first)
- Responsive design for desktop and mobile
