"""
Create a lightweight HTML player for a folder of PNG frames.

Why:
- Works without ffmpeg / pillow / imageio.
- Useful for quickly previewing animations (e.g., trajectory bundles) in browser.

Usage:
  python -m src.visualization.make_html_animation \
    --frames_dir essay/figures/stage_cfg/anim/anim_cfg_bundle_frames_case122 \
    --fps 6

This writes `anim.html` into the frames directory by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _sorted_frames(frames_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir} with pattern '{pattern}'")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="frame_*.png")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--title", type=str, default="Animation preview")
    parser.add_argument("--out_html", type=str, default=None, help="Default: <frames_dir>/anim.html")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    frames = _sorted_frames(frames_dir, str(args.pattern))
    out_html = Path(args.out_html) if args.out_html else (frames_dir / "anim.html")

    # If html is written inside frames_dir, we can reference frames by filename only.
    # Otherwise, use a relative prefix.
    rel_prefix = ""
    try:
        out_parent = out_html.parent.resolve()
        frames_parent = frames_dir.resolve()
        if out_parent != frames_parent:
            rel_prefix = str(frames_parent.relative_to(out_parent)).replace("\\", "/") + "/"
    except Exception:
        rel_prefix = str(frames_dir).replace("\\", "/") + "/"

    frame_names = [rel_prefix + f.name for f in frames]
    fps = max(1, int(args.fps))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{args.title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #111; color: #eee; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    .row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 12px; }}
    button {{ padding: 8px 12px; border-radius: 8px; border: 0; background: #2b6cb0; color: #fff; cursor: pointer; }}
    button:disabled {{ background: #444; cursor: not-allowed; }}
    input[type=range] {{ width: 320px; }}
    .meta {{ opacity: 0.8; font-size: 14px; }}
    .frame {{ background: #000; border-radius: 12px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="row">
      <button id="btn">Play</button>
      <input id="slider" type="range" min="0" max="{len(frame_names)-1}" value="0" step="1"/>
      <span class="meta">FPS: <b>{fps}</b> | Frame: <b id="idx">0</b> / {len(frame_names)-1}</span>
    </div>
    <div class="frame">
      <img id="img" src="{frame_names[0]}" alt="animation frame"/>
    </div>
  </div>
  <script>
    const frames = {frame_names};
    const fps = {fps};
    const img = document.getElementById('img');
    const slider = document.getElementById('slider');
    const idxText = document.getElementById('idx');
    const btn = document.getElementById('btn');
    let timer = null;

    function setFrame(i) {{
      const idx = Math.max(0, Math.min(frames.length-1, i));
      slider.value = idx;
      idxText.textContent = idx;
      img.src = frames[idx];
    }}

    slider.addEventListener('input', (e) => {{
      setFrame(parseInt(e.target.value));
    }});

    function play() {{
      if (timer) return;
      btn.textContent = 'Pause';
      timer = setInterval(() => {{
        const i = (parseInt(slider.value) + 1) % frames.length;
        setFrame(i);
      }}, Math.round(1000 / fps));
    }}

    function pause() {{
      if (!timer) return;
      clearInterval(timer);
      timer = null;
      btn.textContent = 'Play';
    }}

    btn.addEventListener('click', () => {{
      if (timer) pause(); else play();
    }});
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] wrote {out_html}")


if __name__ == "__main__":
    main()

