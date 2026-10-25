"""Synthesize original narration with macOS say, measure it, and author timed clips."""

import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
VOICE = "Samantha"
RATE = 155
SAMPLE_RATE = 24000
CHUNK_GAP = 0.16
LEAD = 0.45
TAIL = 0.75


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    scenes = json.loads((ROOT / "narration.json").read_text())
    cache = ROOT / ".audio-cache"
    cache.mkdir(exist_ok=True)
    (ROOT / "src/narration").mkdir(exist_ok=True)
    (ROOT / "public/narration").mkdir(exist_ok=True)
    metadata = []
    transcript = [
        "# GRPO and online RL: the 2048 explainer",
        "",
        "Original teaching script. Numerical illustrations are not measured training results.",
        "",
    ]
    for scene in scenes:
        combined = bytearray(b"\0" * round(LEAD * SAMPLE_RATE) * 2)
        captions = []
        for phrase in scene["phrases"]:
            # Generate short phrase clips so captions have measured boundaries.
            # Each complete phrase is a caption page, not guessed word alignment.
            spoken = (
                phrase.replace("GRPO", "G R P O")
                .replace("SFT", "S F T")
                .replace("TRL", "T R L")
                .replace("RL", "R L")
                .replace("LoRA", "low rah")
                .replace("KL", "K L")
            )
            signature = hashlib.sha256(f"{VOICE}|{RATE}|{spoken}".encode()).hexdigest()[:20]
            text_path = cache / f"{signature}.txt"
            aiff_path = cache / f"{signature}.aiff"
            wav_path = cache / f"{signature}.wav"
            if not wav_path.exists():
                text_path.write_text(spoken)
                run(
                    [
                        "say",
                        "-v",
                        VOICE,
                        "-r",
                        str(RATE),
                        "-f",
                        str(text_path),
                        "-o",
                        str(aiff_path),
                    ]
                )
                run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(aiff_path),
                        "-ar",
                        str(SAMPLE_RATE),
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(wav_path),
                    ]
                )
            with wave.open(str(wav_path), "rb") as source:
                pcm = source.readframes(source.getnframes())
            start = len(combined) / 2 / SAMPLE_RATE
            combined.extend(pcm)
            end = len(combined) / 2 / SAMPLE_RATE
            captions.append(
                {
                    "text": phrase,
                    "startMs": round(start * 1000),
                    "endMs": round(end * 1000),
                    "timestampMs": None,
                    "confidence": None,
                    "pageBreakAfter": True,
                }
            )
            combined.extend(b"\0" * round(CHUNK_GAP * SAMPLE_RATE) * 2)
        combined.extend(b"\0" * round(TAIL * SAMPLE_RATE) * 2)
        duration = len(combined) / 2 / SAMPLE_RATE
        frames = math.ceil(duration * FPS)
        combined.extend(b"\0" * max(0, round(frames / FPS * SAMPLE_RATE) * 2 - len(combined)))
        wav = cache / f"{scene['id']}.wav"
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(combined)
        target = ROOT / "public/narration" / f"{scene['id']}.mp3"
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav),
                "-af",
                "loudnorm=I=-18:TP=-2:LRA=8",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(target),
            ]
        )
        caption_json = json.dumps(captions, indent=2)
        component = f'''import {{ Audio }} from "@remotion/media";
import {{ staticFile, useVideoConfig }} from "remotion";
import {{ BasicCaptions }} from "../BasicCaptions";

export const {scene["id"]}Narration = () => {{
  const {{fps}} = useVideoConfig();
  return <>
    <Audio name="{scene["title"]} narration" src={{staticFile("narration/{scene["id"]}.mp3")}} premountFor={{fps}} />
    <BasicCaptions width={{1696}} style={{{{position: "absolute", bottom: 55, left: 112}}}} captions={{{caption_json}}} />
  </>;
}};
'''
        (ROOT / "src/narration" / f"{scene['id']}Narration.tsx").write_text(component)
        metadata.append(
            {
                "id": scene["id"],
                "title": scene["title"],
                "frames": frames,
                "duration_seconds": frames / FPS,
                "phrases": captions,
            }
        )
        transcript += [f"## {scene['title']}", "", " ".join(scene["phrases"]), ""]
        print(f"{scene['id']}: {frames / FPS:.1f}s", flush=True)
    total = sum(item["frames"] for item in metadata)
    (ROOT / "public/timing.json").write_text(
        json.dumps(
            {
                "fps": FPS,
                "voice": VOICE,
                "rate": RATE,
                "total_frames": total,
                "duration_seconds": total / FPS,
                "scenes": metadata,
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / "TRANSCRIPT.md").write_text("\n".join(transcript))
    imports = "\n".join(
        f'import {{ {s["id"]}Scene }} from "./scenes/{s["id"]}Scene";' for s in metadata
    )
    sequence_nodes = "\n".join(
        f'      <Series.Sequence name="{s["title"]}" durationInFrames={{{s["frames"]}}} premountFor={{fps}}><{s["id"]}Scene /></Series.Sequence>'
        for s in metadata
    )
    (ROOT / "src/Main.tsx").write_text(
        'import {Series, useVideoConfig} from "remotion";\n'
        + imports
        + "\n\nexport const Explainer = () => {\n  const {fps} = useVideoConfig();\n  return <Series>\n"
        + sequence_nodes
        + "\n  </Series>;\n};\n"
    )
    registrations = "\n".join(
        f'      <Composition id="{s["id"]}" component={{{s["id"]}Scene}} width={{1920}} height={{1080}} fps={{30}} durationInFrames={{{s["frames"]}}} />'
        for s in metadata
    )
    (ROOT / "src/Root.tsx").write_text(
        'import {Composition, Folder} from "remotion";\nimport {Explainer} from "./Main";\n'
        + imports
        + "\n\nexport const RemotionRoot = () => <>\n"
        + f'  <Composition id="GRPO-2048" component={{Explainer}} width={{1920}} height={{1080}} fps={{30}} durationInFrames={{{total}}} />\n'
        + '  <Folder name="Chapters">\n'
        + registrations
        + "\n  </Folder>\n</>;\n"
    )
    print(
        f"Total: {total / FPS:.1f}s ({sum(len(p.split()) for s in scenes for p in s['phrases'])} spoken words)"
    )


if __name__ == "__main__":
    main()
