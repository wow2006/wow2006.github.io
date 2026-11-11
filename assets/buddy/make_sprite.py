# Slices the character sheet (icons.png at the repo root) into one horizontal
# sprite strip, static/buddy/sprite.webp, used by static/js/buddy.js.
# Run: uv run --with pillow --with numpy python assets/buddy/make_sprite.py
# Frame order must match FRAMES in static/js/buddy.js.
import numpy as np
from PIL import Image

SCALE = 0.75  # displayed at ~120 px tall, so this stays sharp on 2x screens
# Row bands below the baked-in labels, and the column x-ranges of each pose.
ROWS = [
    ((85, 409), [(400, 540), (573, 711), (748, 894), (930, 1073), (1113, 1278)]),  # skip the portrait
    ((476, 800), [(16, 226), (250, 458), (492, 670), (698, 902), (923, 1086), (1109, 1281)]),
    ((857, 1213), [(13, 172), (197, 435), (479, 657), (663, 903), (909, 1069), (1094, 1281)]),
]

sheet = Image.open("icons.png").convert("RGBA")
frames = []
for (y0, y1), cols in ROWS:
    for x0, x1 in cols:
        f = sheet.crop((x0, y0, x1, y1))
        f = f.crop(f.getbbox())
        frames.append(f.resize((round(f.width * SCALE), round(f.height * SCALE)), Image.LANCZOS))
w = max(f.width for f in frames)
h = max(f.height for f in frames)
strip = Image.new("RGBA", (w * len(frames), h))
for i, f in enumerate(frames):  # feet on the bottom edge, centred in the cell
    strip.paste(f, (i * w + (w - f.width) // 2, h - f.height))
strip.save("static/buddy/sprite.webp", quality=90, method=6)
print(len(frames), "frames, cell", w, "x", h)
