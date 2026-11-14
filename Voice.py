import asyncio, collections, glob, io, json, os, re, sys

import edge_tts
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

VOICE = os.environ.get("VOICE") or "ar-SA-HamedNeural"
RATE = os.environ.get("RATE") or "-10%"
VARIANTS = os.environ.get("VARIANTS") or "first"   # first: из «А / Б» только А; both: оба
BITRATE = os.environ.get("BITRATE") or "24k"
PARALLEL = int(os.environ.get("PARALLEL") or "4")
SR = 24000
OUT = "out"


def norm(t):
    """Текст для озвучки: без невидимых символов и татвиля, один вариант из «А / Б»."""
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]", "", t).replace("\u00a0", " ").strip()
    parts = [p.strip() for p in re.split(r"\s*/\s*", t) if p.strip()]
    if parts:
        t = parts[0] if VARIANTS == "first" else "، ".join(parts)
    t = re.sub(r"\s*ـ{2,}\s*", " ، ", t)
    t = t.replace("ـ", "")
    t = re.sub(r"[()‘’“”\"«»]", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ،")
    return t


def load_groups():
    files = sorted(glob.glob("*.html"))
    if not files:
        sys.exit("Не найден .html в корне репозитория")
    html = open(files[0], encoding="utf-8").read()
    ALL = json.loads(re.search(r"let ALL=(\[.*?\]);\n", html, re.S).group(1))
    groups = collections.OrderedDict()
    for i, x in enumerate(ALL):
        key = norm(x[1])
        if key:
            groups.setdefault(key, []).append(i)
    print(f"Файл: {files[0]}; фраз: {len(ALL)}; уникальных текстов: {len(groups)}", flush=True)
    return ALL, groups


async def synth(text):
    try:
        comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
    except TypeError:
        comm = edge_tts.Communicate(text, VOICE, rate=RATE)
    audio, words = bytearray(), []
    async for ch in comm.stream():
        if ch["type"] == "audio":
            audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            words.append([round(ch["offset"] / 1e7, 3), round(ch["duration"] / 1e7, 3), ch["text"]])
    if not audio:
        raise RuntimeError("пустой ответ")
    return bytes(audio), words


def finish(data, words):
    seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    lead = detect_leading_silence(seg, silence_threshold=-45)
    tail = detect_leading_silence(seg.reverse(), silence_threshold=-45)
    a = max(0, lead - 50)
    b = len(seg) - max(0, tail - 120)
    seg = seg[a:b].set_channels(1).set_frame_rate(SR).fade_out(40)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=BITRATE)
    words = [[max(0.0, round(s - a / 1000, 3)), d, w] for s, d, w in words]
    return buf.getvalue(), words, len(seg) / 1000


async def main():
    os.makedirs(OUT, exist_ok=True)
    ALL, groups = load_groups()
    META, failed = {}, []
    sem = asyncio.Semaphore(PARALLEL)
    state = {"n": 0}

    async def job(text, ids):
        err = None
        async with sem:
            for k in range(6):
                try:
                    data, words = await synth(text)
                    break
                except Exception as e:
                    err = e
                    await asyncio.sleep(2 * (k + 1))
            else:
                failed.append((ids[0], text, repr(err)))
                return
        mp3, words, dur = await asyncio.to_thread(finish, data, words)
        fn = f"{ids[0]:04d}.mp3"
        with open(f"{OUT}/{fn}", "wb") as f:
            f.write(mp3)
        META[text] = {"file": fn, "dur": round(dur, 2), "words": words}
        state["n"] += 1
        if state["n"] % 100 == 0:
            print(f"  готово {state['n']} / {len(groups)}", flush=True)

    await asyncio.gather(*[job(t, ids) for t, ids in groups.items()])

    manifest = {"voice": VOICE, "rate": RATE, "variants": VARIANTS, "items": {}, "files": {}}
    for text, ids in groups.items():
        m = META.get(text)
        if not m:
            continue
        manifest["files"][m["file"]] = {"text": text, "dur": m["dur"], "words": m["words"]}
        for i in ids:
            manifest["items"][str(i)] = m["file"]
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    mb = sum(os.path.getsize(f"{OUT}/{n}") for n in os.listdir(OUT)) / 1048576
    print(f"Фраз с озвучкой: {len(manifest['items'])} из {len(ALL)}; файлов: {len(manifest['files'])}; {mb:.1f} МБ")
    if failed:
        print(f"ОШИБОК: {len(failed)}")
        for x in failed[:20]:
            print(" ", x)
        sys.exit(1)


asyncio.run(main())
