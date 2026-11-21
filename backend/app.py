import os, re, json, tempfile, subprocess, zipfile, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests

app = FastAPI(title="Mayank AI Clip Maker API")

ALLOWED_ORIGINS = [
    "https://mayank18-prog.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_UPLOAD = 150 * 1024 * 1024
MAX_AUDIO_FOR_GROQ = 24 * 1024 * 1024

@app.get("/")
def health():
    return {"ok": True, "service": "Mayank AI Clip Maker API"}

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "Media processing failed")[-1800:])
    return p.stdout

def ffprobe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())

def choose_ranges(words, duration, count, min_len, max_len):
    # Select high-information transcript windows, then avoid overlapping clips.
    if not words:
        raise ValueError("Speech was not detected clearly enough to create captioned clips.")
    candidates = []
    step = max(5.0, min_len / 3)
    t = 0.0
    while t < duration:
        end = min(duration, t + max_len)
        chunk = [w for w in words if t <= float(w.get("start", 0)) < end]
        if chunk:
            text = " ".join(str(w.get("word", "")) for w in chunk)
            density = len(chunk) / max(1.0, end - t)
            punctuation = sum(text.count(x) for x in ".!?।")
            # Prefer speech-rich windows with a natural ending.
            score = density + min(punctuation, 5) * 0.25
            candidates.append((score, t, end))
        t += step
    candidates.sort(reverse=True)
    chosen = []
    for score, start, end in candidates:
        if all(end <= a or start >= b for a, b in chosen):
            chosen.append((start, end))
            if len(chosen) >= count:
                break
    if not chosen:
        chosen = [(0.0, min(duration, max_len))]
    return sorted(chosen)

def ass_time(seconds):
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"

def esc_ass(s):
    return str(s).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

def make_ass(words, start, end, output):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,66,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    current = [w for w in words if float(w.get("end", 0)) > start and float(w.get("start", 0)) < end]
    # Group a few words per subtitle line; karaoke highlight advances word-by-word.
    for i in range(0, len(current), 5):
        group = current[i:i+5]
        if not group:
            continue
        st = max(start, float(group[0].get("start", start))) - start
        en = min(end, float(group[-1].get("end", end))) - start
        parts = []
        for w in group:
            a = max(0.03, float(w.get("end", 0)) - float(w.get("start", 0)))
            centis = max(1, int(a * 100))
            parts.append(r"{\kf" + str(centis) + "}" + esc_ass(w.get("word", "")))
        lines.append(f"Dialogue: 0,{ass_time(st)},{ass_time(max(st+0.2,en))},Default,,0,0,0,,{' '.join(parts)}\n")
    Path(output).write_text("".join(lines), encoding="utf-8")

@app.post("/api/generate")
async def generate(
    video: UploadFile = File(...),
    clip_count: int = Form(3),
    clip_length: str = Form("15-30"),
    video_format: str = Form("9:16 Shorts"),
    caption_language: str = Form("hinglish"),
):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(503, "Backend setup incomplete: GROQ_API_KEY secret is missing.")
    if clip_count not in (1, 3, 5, 8, 10):
        raise HTTPException(400, "Choose a supported clip count.")
    if clip_length not in ("15-30", "30-60", "60-90"):
        raise HTTPException(400, "Invalid clip length.")
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
        raise HTTPException(400, "Please upload an MP4, MOV, MKV, WEBM or M4V video.")
    raw = await video.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Video is too large for this starter version (150 MB maximum).")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / ("input" + suffix)
        src.write_bytes(raw)
        try:
            duration = ffprobe_duration(src)
            audio = td / "audio.mp3"
            run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                 "-b:a", "48k", str(audio)])
            if audio.stat().st_size > MAX_AUDIO_FOR_GROQ:
                raise HTTPException(413, "Audio track is over the speech API's 24 MB upload limit. Try a shorter video.")
            with audio.open("rb") as f:
                response = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": ("audio.mp3", f, "audio/mpeg")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                    },
                    timeout=240,
                )
            if response.status_code != 200:
                raise HTTPException(502, "Speech transcription service error: " + response.text[:500])
            transcript = response.json()
            words = transcript.get("words", [])
            if not words:
                raise HTTPException(422, "No word timestamps returned. Try another video with clear speech.")
            lo, hi = map(int, clip_length.split("-"))
            ranges = choose_ranges(words, duration, clip_count, lo, hi)
            outdir = td / "clips"
            outdir.mkdir()
            created = []
            for idx, (start, end) in enumerate(ranges, 1):
                # Keep requested range within source duration.
                end = min(end, duration)
                ass = td / f"clip{idx}.ass"
                make_ass(words, start, end, ass)
                mp4 = outdir / f"mayank_clip_{idx}.mp4"
                vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" if video_format == "9:16 Shorts" else (
                    "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080" if video_format == "1:1 Square" else
                    "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                )
                # ASS subtitles are burned in; karaoke timing highlights the spoken word.
                ass_path = str(ass).replace(":", r"\\:")
                vf += ",ass=" + ass_path
                run(["ffmpeg", "-y", "-ss", str(start), "-i", str(src), "-t", str(end-start),
                     "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                     "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(mp4)])
                created.append(mp4)
            zip_path = td / "mayank_clips.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in created:
                    z.write(p, p.name)
            data = zip_path.read_bytes()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, "Processing failed: " + str(e)[:500])
    from io import BytesIO
    return StreamingResponse(BytesIO(data), media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="mayank_clips.zip"'})
