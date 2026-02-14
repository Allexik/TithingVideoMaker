# Tithing Video Maker

Tool that renders an animated tithing video over a provided loopable background video.
You can run it in CLI mode or in a simple windowed UI mode.

## What it builds

The output video has two animated parts:

1. Verse part:
   - Text (centered):  
     `Нехай кожен дає, як серце йому призволяє, не в смутку й не з примусу, бо Бог любить того, хто з радістю дає!`
   - Reference below it: `2 Коринтян 9:7`

2. Money part:
   - Left: animated donut chart (`collected / target`), percentage inside, month label above, numbers below.
   - Optional overhead amount (>100%) is highlighted with an extra red outer arc.
   - Right: QR image + `Donate ❤` below.

All animations are automatically scaled to the exact background video duration.

## Install

```bash
poetry install
```

Or with pip:

```bash
pip install moviepy numpy Pillow pymupdf
```

MoviePy typically uses bundled `imageio-ffmpeg`, so a system FFmpeg install is usually not required.
If your environment cannot use the bundled binary, install FFmpeg and make it available on PATH.

## Usage (CLI)

```bash
python main.py \
  --target 50000 \
  --collected 32750 \
  --month 1 \
  --qr "assets/qr.svg" \
  --output "output/tithing_january.mp4"
```

Short flags are also supported: `-t -c -m -b -o -q -f`.

### Arguments

- `--target` (required): monthly target amount
- `--collected` (required): collected amount
- `--month` (required): month number in range `1..12`
- `--background` (optional): path to background video (`backgrounds/background.mp4` by default)
- `--qr` (optional): path to QR image (`assets/qr.svg` by default), file must exist
- `--output` (optional): output mp4 path (`output/tithing_video.mp4` by default)
- `--fps` (optional): output frame rate, default `60`
- `--ui` (optional): launch windowed form mode instead of CLI args

## Usage (UI)

```bash
python main.py --ui
```

In UI mode, enter the same values (`target`, `collected`, `month`, `background`, `qr`, `output`, `fps`) and click **Render**.
