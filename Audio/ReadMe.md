## Preview UI (local) — step-by-step

### 0) Put files in one folder

- make_preview.py
- your_chart.json (example: brandy-100.json)
- (optional) preview template is generated automatically

### 1) Generate / update preview.html

From the folder:
python3 make_preview.py

Notes:

- If your script has a JSON_PATH variable, set it to your file name (ex: "brandy-100.json")
- If your script supports CLI args, you can do:
  python3 make_preview.py path/to/brandy-100.json

### 2) Host locally over HTTP (IMPORTANT for YouTube)

In the same folder (where preview.html is):
python3 -m http.server 8000

Keep this terminal running.

### 3) Open the preview in your browser

Go to:
http://localhost:8000/preview.html

DO NOT open preview.html with file://
(YouTube iframe will error if you do)

### 4) Refresh workflow (edit JSON -> re-generate -> refresh)

Whenever you change the JSON:

1. Re-run:
   python3 make_preview.py
2. Refresh the browser tab:
   Cmd+R (Mac) / Ctrl+R (Windows/Linux)
   If it looks cached, hard refresh:
   Cmd+Shift+R / Ctrl+Shift+R

### 5) Common issues

- "Watch video on YouTube" / Error 153:
  You opened the file with file:// or you are not serving over HTTP.
  Fix: use http://localhost:8000/preview.html

- Page loads but boxes don't flash:
  Make sure preview.html and the JSON file are in the same folder (or the path in make_preview.py is correct).
  Check browser console for 404.

### 6) Optional: change port

python3 -m http.server 3000
then open:
http://localhost:3000/preview.html
