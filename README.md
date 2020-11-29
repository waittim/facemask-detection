# Real-time In-browser Face Mask Detection

### Method 1: github pages(chrome)

1. launch google chrome browser, open chrome://flags and enable all experimental webassembly features

2. re-launch google chrome browser, open https://facemask-detection.com/test.html and enjoy

3. re-launch google chrome browser, open https://facemask-detection.com/index.html and enjoy

### Method 2: run locally(chrome)

1. start a http server
```
python3 -m http.server 8888
```

2. launch google chrome browser, open chrome://flags and enable all experimental webassembly features

3. re-launch google chrome browser, open http://127.0.0.1:8888/test.html and enjoy

4. re-launch google chrome browser, open http://127.0.0.1:8888/index.html and enjoy

### Method 3: run locally(firefox-nightly)

1. start a http server
```
python3 server.py
```

2. download firefox-nightly from https://www.mozilla.org/en-US/firefox/channel/desktop

3. launch firefox-nightly browser, open http://127.0.0.1:8888/test.html and enjoy

4. launch firefox-nightly browser, open http://127.0.0.1:8888/index.html and enjoy
