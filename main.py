import os
import tempfile
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# audio_chord_detector importado solo cuando se reactive (fase futura)
# from audio_chord_detector import detect_chord_timeline

# ─── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(title="Easy Chords IA - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Mapeo de notas ──────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_NAMES_ES = {
    "C": "Do", "C#": "Do#", "D": "Re", "D#": "Re#",
    "E": "Mi", "F": "Fa", "F#": "Fa#", "G": "Sol",
    "G#": "Sol#", "A": "La", "A#": "La#", "B": "Si",
}

KEY_NAMES_ES = {
    "C major": "Do mayor", "C# major": "Do# mayor", "D major": "Re mayor",
    "D# major": "Re# mayor", "E major": "Mi mayor", "F major": "Fa mayor",
    "F# major": "Fa# mayor", "G major": "Sol mayor", "G# major": "Sol# mayor",
    "A major": "La mayor", "A# major": "La# mayor", "B major": "Si mayor",
    "C minor": "Do menor", "C# minor": "Do# menor", "D minor": "Re menor",
    "D# minor": "Re# menor", "E minor": "Mi menor", "F minor": "Fa menor",
    "F# minor": "Fa# menor", "G minor": "Sol menor", "G# minor": "Sol# menor",
    "A minor": "La menor", "A# minor": "La# menor", "B minor": "Si menor",
}

# Perfiles de Krumhansl para detección de tonalidad
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ─── Funciones de análisis ────────────────────────────────────────────────────

def detect_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Detecta la tonalidad usando correlación con perfiles de Krumhansl."""
    major_scores = []
    minor_scores = []

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        major_scores.append(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        minor_scores.append(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] >= minor_scores[best_minor]:
        key_en = f"{NOTE_NAMES[best_major]} major"
    else:
        key_en = f"{NOTE_NAMES[best_minor]} minor"

    key_es = KEY_NAMES_ES.get(key_en, key_en)
    return key_en, key_es


def chroma_to_chord(chroma: np.ndarray) -> str:
    """Convierte un vector de chroma a un acorde simple."""
    # Plantillas de acordes (mayor y menor para las 12 notas)
    templates = {}
    for i, note in enumerate(NOTE_NAMES):
        # Mayor: raíz, tercera mayor, quinta
        major = np.zeros(12)
        major[i % 12] = 1
        major[(i + 4) % 12] = 1
        major[(i + 7) % 12] = 1
        templates[note] = major

        # Menor: raíz, tercera menor, quinta
        minor = np.zeros(12)
        minor[i % 12] = 1
        minor[(i + 3) % 12] = 1
        minor[(i + 7) % 12] = 1
        templates[f"{note}m"] = minor

    best_chord = "C"
    best_score = -1

    for chord_name, template in templates.items():
        score = np.dot(chroma / (np.linalg.norm(chroma) + 1e-6), template / (np.linalg.norm(template) + 1e-6))
        if score > best_score:
            best_score = score
            best_chord = chord_name

    # Mapear a acordes básicos que un principiante puede tocar
    SIMPLIFY = {
        "C#": "D", "Db": "D", "D#": "E", "Eb": "E",
        "F#": "G", "Gb": "G", "G#": "G", "Ab": "G",
        "A#": "A", "Bb": "A", "B": "C",
        "C#m": "Bm", "Dbm": "Bm", "D#m": "Dm", "Ebm": "Dm",
        "F#m": "Em", "Gbm": "Em", "G#m": "Am", "Abm": "Am",
        "A#m": "Am", "Bbm": "Am", "Bm": "Bm",
    }

    return SIMPLIFY.get(best_chord, best_chord)


def get_notes_from_key(key_en: str) -> list[str]:
    """Devuelve las notas de la tonalidad en español."""
    SCALES = {
        "C major":  ["C", "D", "E", "F", "G", "A", "B"],
        "G major":  ["G", "A", "B", "C", "D", "E", "F#"],
        "D major":  ["D", "E", "F#", "G", "A", "B", "C#"],
        "A major":  ["A", "B", "C#", "D", "E", "F#", "G#"],
        "E major":  ["E", "F#", "G#", "A", "B", "C#", "D#"],
        "F major":  ["F", "G", "A", "A#", "C", "D", "E"],
        "A minor":  ["A", "B", "C", "D", "E", "F", "G"],
        "E minor":  ["E", "F#", "G", "A", "B", "C", "D"],
        "D minor":  ["D", "E", "F", "G", "A", "A#", "C"],
        "G minor":  ["G", "A", "A#", "C", "D", "D#", "F"],
        "B minor":  ["B", "C#", "D", "E", "F#", "G", "A"],
        "C# major": ["C#", "D#", "F", "F#", "G#", "A#", "C"],
    }

    notes_en = SCALES.get(key_en, ["C", "D", "E", "F", "G", "A", "B"])
    return [NOTE_NAMES_ES.get(n, n) for n in notes_en]


def simplify_progression(chords: list[str]) -> list[str]:
    """Elimina duplicados consecutivos y limita a 8 acordes únicos representativos."""
    seen = []
    for chord in chords:
        if not seen or chord != seen[-1]:
            seen.append(chord)

    # Tomar máximo 8 acordes únicos
    unique = list(dict.fromkeys(seen))
    return unique[:8]


def get_key_chords(key_en: str) -> list[str]:
    """
    Devuelve los acordes diatónicos de la tonalidad, ordenados por importancia.
    Estos son los acordes que realmente se usan en esa tonalidad.
    """
    KEY_CHORDS = {
        # Mayores: I, IV, V, vi, ii, iii, VII
        "C major":  ["C", "F", "G", "Am", "Dm", "Em", "Bm"],
        "G major":  ["G", "C", "D", "Em", "Am", "Bm", "F"],
        "D major":  ["D", "G", "A", "Bm", "Em", "F", "C"],
        "A major":  ["A", "D", "E", "F", "Bm", "Em", "G"],
        "E major":  ["E", "A", "B", "C", "F", "Bm", "Am"],
        "F major":  ["F", "Bb", "C", "Dm", "Gm", "Am", "Em"],
        "Bb major": ["Bb", "Eb", "F", "Gm", "Cm", "Dm", "Am"],
        # Menores: i, iv, v, III, VI, VII, ii
        "A minor":  ["Am", "Dm", "Em", "C", "F", "G", "Bm"],
        "E minor":  ["Em", "Am", "Bm", "G", "C", "D", "F"],
        "D minor":  ["Dm", "Gm", "Am", "F", "Bb", "C", "Em"],
        "G minor":  ["Gm", "Cm", "Dm", "Bb", "Eb", "F", "Am"],
        "B minor":  ["Bm", "Em", "F", "D", "G", "A", "Dm"],
        "C minor":  ["Cm", "Fm", "Gm", "Eb", "Ab", "Bb", "Dm"],
        "F minor":  ["Fm", "Bb", "Cm", "Ab", "Db", "Eb", "Gm"],
        "C# minor": ["Bm", "Em", "F", "D", "G", "A", "Am"],
    }
    # Filtrar solo los que tenemos diagramas
    KNOWN = {"C","D","E","F","G","A","B","Am","Bm","Cm","Dm","Em","Fm","Gm"}
    chords = KEY_CHORDS.get(key_en, ["C", "F", "G", "Am"])
    return [c for c in chords if c in KNOWN][:6]


def build_timeline_from_key(y: np.ndarray, sr: int, key_en: str) -> tuple[list[dict], list[str]]:
    """
    Detecta secciones de la canción por energía (intro, estrofa, coro, etc.)
    y asigna los acordes diatónicos de la tonalidad en orden cíclico.
    Mucho más confiable que intentar detectar acordes exactos.
    """
    from collections import Counter

    key_chords = get_key_chords(key_en)
    if not key_chords:
        return [], []

    # Detectar beats para timing
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Calcular energía RMS por segmento para detectar secciones
    frame_length = int(sr * 2)  # segmentos de 2 segundos
    hop = frame_length // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Detectar cambios de sección por variaciones de energía
    # Normalizar RMS
    rms_norm = rms / (rms.max() + 1e-6)

    # Agrupar en secciones de ~8 segundos (típico de un verso/coro)
    section_duration = 8.0
    n_sections = max(1, int(times[-1] / section_duration))

    timeline = []
    chord_idx = 0
    seen_last = None

    for i in range(n_sections):
        t_start = i * section_duration
        minutes = int(t_start // 60)
        secs = int(t_start % 60)

        # Asignar acorde cíclicamente desde los acordes de la tonalidad
        chord = key_chords[chord_idx % len(key_chords)]
        chord_idx += 1

        if chord != seen_last:
            timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
            seen_last = chord

    # Progresión: los primeros 4-5 acordes únicos de la tonalidad
    progression = list(dict.fromkeys(key_chords[:5]))

    return timeline[:16], progression


def build_timeline(chords_per_segment: list[str], hop_length: int, sr: int, segment_seconds: float) -> list[dict]:
    """Construye la timeline con timestamps (fallback)."""
    timeline = []
    seen_last = None

    for i, chord in enumerate(chords_per_segment):
        if chord != seen_last:
            seconds = i * segment_seconds
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            timeline.append({
                "time": f"{minutes}:{secs:02d}",
                "chord": chord,
            })
            seen_last = chord

    return timeline[:12]


# ─── Búsqueda de acordes reales desde fuentes externas ───────────────────────

SIMPLIFY_CHORD = {
    "C#": "D", "Db": "D", "D#": "E", "Eb": "E",
    "F#": "G", "Gb": "G", "G#": "G", "Ab": "G",
    "A#": "A", "Bb": "A", "B": "C",
    "C#m": "Bm", "Dbm": "Bm", "D#m": "Dm", "Ebm": "Dm",
    "F#m": "Em", "Gbm": "Em", "G#m": "Am", "Abm": "Am",
    "A#m": "Am", "Bbm": "Am",
    # Séptimas → base
    "A7": "A", "B7": "C", "C7": "C", "D7": "D", "E7": "E", "F7": "F", "G7": "G",
    "Am7": "Am", "Bm7": "Bm", "Dm7": "Dm", "Em7": "Em", "Fm7": "Fm",
    "Amaj7": "A", "Cmaj7": "C", "Dmaj7": "D", "Emaj7": "E", "Fmaj7": "F", "Gmaj7": "G",
    "Asus2": "A", "Asus4": "A", "Dsus2": "D", "Dsus4": "D", "Esus4": "E",
    "Gsus2": "G", "Gsus4": "G", "Cadd9": "C", "Gadd9": "G",
}

# ── Acordes válidos: se construyen dinámicamente desde canciones_curadas.json ──
import json as _json_meta
import os as _os_meta

def _load_known_chords() -> set:
    """
    Lee acordes_permitidos_base de canciones_curadas.json y construye
    el set de acordes válidos. Si el archivo no existe, usa un set mínimo.
    """
    candidates = [
        _os_meta.path.join(_os_meta.path.dirname(__file__), "canciones_curadas.json"),
        _os_meta.path.join(_os_meta.path.dirname(__file__), "..", "App", "canciones_curadas.json"),
    ]
    for path in candidates:
        if _os_meta.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    db = _json_meta.load(f)
                base = db.get("_meta", {}).get("acordes_permitidos_base", {})
                skip = {"simplificar_si_aparecen"}
                known = set()
                for key, values in base.items():
                    if key not in skip and isinstance(values, list):
                        known.update(values)
                if known:
                    print(f"KNOWN_CHORDS: {len(known)} acordes cargados desde {path}")
                    return known
            except Exception as e:
                print(f"KNOWN_CHORDS: error leyendo JSON ({e}), usando set minimo")

    # fallback minimo si no encuentra el archivo
    return {"C","D","E","F","G","A","B","Am","Bm","Cm","Dm","Em","Fm","Gm",
            "A7","B7","C7","D7","E7","F7","G7","F#m","C#m","G#m",
            "Asus4","Dsus4","Esus4","Asus2","Dsus2"}

KNOWN_CHORDS = _load_known_chords()


def _get_simplify_criteria() -> list:
    """Lee criterios de simplificacion desde canciones_curadas.json."""
    candidates = [
        _os_meta.path.join(_os_meta.path.dirname(__file__), "canciones_curadas.json"),
        _os_meta.path.join(_os_meta.path.dirname(__file__), "..", "App", "canciones_curadas.json"),
    ]
    for path in candidates:
        if _os_meta.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    db = _json_meta.load(f)
                return db.get("_meta", {}).get("acordes_permitidos_base", {}).get(
                    "simplificar_si_aparecen", []
                )
            except Exception:
                pass
    return ["dim", "aug", "m7b5", "cejilla_compleja", "tensiones_avanzadas"]


def simplify_chord_name(chord: str) -> str:
    """
    - Si el acorde esta en KNOWN_CHORDS → devolver tal cual, sin tocar.
    - Si contiene criterio de simplificar_si_aparecen → simplificar a raiz.
    - En cualquier otro caso → devolver el acorde original sin modificar.
    """
    if not chord:
        return chord

    # Ya es un acorde valido conocido
    if chord in KNOWN_CHORDS:
        return chord

    # Intentar con el diccionario de mapeo existente
    mapped = SIMPLIFY_CHORD.get(chord)
    if mapped and mapped in KNOWN_CHORDS:
        return mapped

    # Verificar si cae en criterios de simplificacion
    criterios = _get_simplify_criteria()
    chord_lower = chord.lower()
    for criterio in criterios:
        if criterio.lower() in chord_lower:
            # Simplificar: extraer raiz y modo (mayor/menor)
            root = chord[0].upper()
            if len(chord) > 1 and chord[1] in ("#", "b"):
                root += chord[1]
                rest = chord[2:]
            else:
                rest = chord[1:]
            if rest.startswith("m") and not rest.startswith("maj"):
                simplified = root + "m"
            else:
                simplified = root
            if simplified in KNOWN_CHORDS:
                return simplified
            # Si la simplificacion tampoco esta en KNOWN_CHORDS, devolver original
            return chord

    # En caso de duda: devolver original sin modificar
    return chord

def _normalize_for_match(s: str) -> str:
    """Normaliza texto para comparacion: minusculas, sin tildes, sin espacios extra."""
    import unicodedata
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = " ".join(s.split())
    return s


def find_curated_song(artist: str, title: str) -> dict | None:
    """
    Busca en canciones_curadas.json una coincidencia por artista + titulo.
    Normaliza texto antes de comparar (tildes, mayusculas, espacios).
    Devuelve la entrada completa de la cancion o None si no encuentra.
    """
    import os, json as _json

    candidates = [
        os.path.join(os.path.dirname(__file__), "canciones_curadas.json"),
        os.path.join(os.path.dirname(__file__), "..", "App", "canciones_curadas.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    db = _json.load(f)

                artist_norm = _normalize_for_match(artist)
                title_norm  = _normalize_for_match(title)

                for cancion in db.get("canciones", []):
                    if (
                        _normalize_for_match(cancion.get("artista", "")) == artist_norm
                        and _normalize_for_match(cancion.get("titulo", "")) == title_norm
                    ):
                        print(f"BASE CURADA: encontrada '{cancion['artista']} - {cancion['titulo']}'")
                        return cancion

                print(f"BASE CURADA: no encontrada '{artist} - {title}'")
                return None

            except Exception as e:
                print(f"BASE CURADA EXCEPTION: {e}")
                return None

    print("BASE CURADA: archivo canciones_curadas.json no encontrado")
    return None


def _get_allowed_chords_from_curated(curated: dict) -> list[str]:
    """
    Extrae la lista de acordes únicos presentes en las secciones de una canción curada.
    Se usa como allowed_chords para guiar al detector automático de audio.
    """
    chords: list[str] = []
    for seccion in curated.get("secciones", []):
        for linea in seccion.get("lineas", []):
            acorde = linea.get("acorde", "").strip()
            if acorde and acorde != "..." and acorde not in chords:
                chords.append(acorde)
        for acorde in seccion.get("acordes_bloque", []):
            if acorde and acorde not in chords:
                chords.append(acorde)
    return chords


def build_from_chord_timeline(chord_timeline: list[dict]) -> tuple[list, list, list]:
    """
    Convierte chord_timeline (array de {time_seconds, chord}) en los tres
    formatos que necesita el endpoint: timeline UI, chordTimeline raw y progresión.
    """
    timeline = []
    for entry in chord_timeline:
        t = entry["time_seconds"]
        minutes = int(t // 60)
        seconds = int(t % 60)
        timeline.append({
            "time": f"{minutes}:{seconds:02d}",
            "chord": entry["chord"],
        })

    chord_timeline_raw = [
        {"time_seconds": e["time_seconds"], "chord": e["chord"]}
        for e in chord_timeline
    ]

    chords_in_order = [e["chord"] for e in chord_timeline]
    deduped = []
    for c in chords_in_order:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    simplified_progression = list(dict.fromkeys(deduped))[:8]

    return timeline, chord_timeline_raw, simplified_progression


def build_lyric_chords_from_curated(curated: dict, synced_lyrics: list[dict]) -> dict | None:
    """
    Construye lyric_chords usando la cancion curada y los timestamps de LRCLIB.
    Distribuye los acordes de cada seccion sobre las lineas de letra correspondientes.
    """
    if not synced_lyrics:
        return None

    secciones = curated.get("secciones", [])
    if not secciones:
        return None

    # Aplanar todos los acordes de todas las secciones en orden
    all_chords = []
    for seccion in secciones:
        # Si la seccion tiene lineas con acorde, usarlas
        lineas = seccion.get("lineas", [])
        bloque = seccion.get("acordes_bloque", [])

        if lineas:
            for linea in lineas:
                acorde = linea.get("acorde", "")
                if acorde and acorde != "...":
                    all_chords.append(acorde)
        elif bloque:
            # Si no hay lineas, repetir el bloque para estimar
            all_chords.extend(bloque)

    if not all_chords:
        return None

    # Mapear acordes a lineas de LRCLIB
    lyric_chords = []
    total_lyrics = len(synced_lyrics)
    total_chords = len(all_chords)

    for i, sl in enumerate(synced_lyrics):
        # Distribuir proporcionalmente
        chord_idx = min(int(i * total_chords / total_lyrics), total_chords - 1)
        chord = all_chords[chord_idx]
        if chord in KNOWN_CHORDS:
            lyric_chords.append({
                "lyric": sl["text"],
                "chord": chord,
                "time_seconds": sl["time_seconds"],
            })
        else:
            # Si el acorde no esta en KNOWN_CHORDS, heredar el anterior
            prev = lyric_chords[-1]["chord"] if lyric_chords else "C"
            lyric_chords.append({
                "lyric": sl["text"],
                "chord": prev,
                "time_seconds": sl["time_seconds"],
            })

    if len(lyric_chords) < 2:
        return None

    # Progresion
    chords_in_order = [lc["chord"] for lc in lyric_chords]
    deduped = []
    for c in chords_in_order:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    main_progression = list(dict.fromkeys(deduped))[:8]

    print(f"BASE CURADA OK: {len(lyric_chords)} lineas mapeadas, progresion={main_progression}")

    return {
        "lyric_chords": lyric_chords,
        "chords": deduped[:20],
        "main_progression": main_progression,
        "source": "curated",
        "capo": curated.get("capo"),
    }


async def fetch_lyrics_from_lrclib(artist: str, title: str) -> list[dict]:
    """
    Busca la letra sincronizada en LRCLIB y la devuelve como lista de
    {"time_seconds": float, "text": str}.
    Retorna lista vacía si no encuentra nada.
    """
    import httpx
    import re

    search_query = f"{artist} {title}".strip()
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://lrclib.net/api/search",
                params={"q": search_query},
                headers={"User-Agent": "EasyChordsIA/1.0"},
                timeout=60,
            )
        if resp.status_code != 200:
            return []

        results = resp.json()
        chosen = next((r for r in results if r.get("syncedLyrics")), None)
        if not chosen:
            return []

        synced_lyrics = []
        for line in chosen["syncedLyrics"].splitlines():
            m = pattern.match(line.strip())
            if m and m.group(3).strip():
                synced_lyrics.append({
                    "time_seconds": int(m.group(1)) * 60 + float(m.group(2)),
                    "text": m.group(3).strip()
                })
        return synced_lyrics

    except Exception as e:
        print(f"LRCLIB EXCEPTION: {e}")
        return []


async def search_cifraclub(artist: str, title: str) -> list[str] | None:
    """
    Busca acordes reales en Cifra Club (mejor para música latina y española).
    Devuelve lista de acordes o None si no encuentra.
    """
    import httpx
    import re

    def to_slug(s: str) -> str:
        s = s.lower().strip()
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n"),("ü","u")]:
            s = s.replace(a, b)
        s = re.sub(r"[^a-z0-9\s-]", "", s)
        s = re.sub(r"\s+", "-", s.strip())
        return s

    artist_slug = to_slug(artist)
    title_slug  = to_slug(title)
    url = f"https://www.cifraclub.com.ar/{artist_slug}/{title_slug}/"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=60,
                follow_redirects=True,
            )

        if resp.status_code != 200:
            print(f"CIFRACLUB: {resp.status_code} para {url}")
            return None

        html = resp.text
        chord_pattern = re.compile(r'<b>([A-G][#b]?(?:m|maj|min|sus|dim|aug|add)?[0-9]?)</b>')
        raw_chords = chord_pattern.findall(html)

        if len(raw_chords) < 4:
            print(f"CIFRACLUB: pocos acordes ({len(raw_chords)}) en {url}")
            return None

        simplified = [simplify_chord_name(c) for c in raw_chords]
        valid = [c for c in simplified if c in KNOWN_CHORDS]

        if len(valid) < 4:
            return None

        print(f"CIFRACLUB OK: {len(valid)} acordes para '{artist} - {title}'")
        return valid

    except Exception as e:
        print(f"CIFRACLUB EXCEPTION: {e}")
        return None


async def search_ultimate_guitar(artist: str, title: str) -> list[str] | None:
    """
    Busca acordes en Ultimate Guitar via su buscador interno.
    Devuelve lista de acordes o None si no encuentra.
    """
    import httpx
    import re
    import json as _json
    import html as html_module

    query = f"{artist} {title}".strip()

    try:
        async with httpx.AsyncClient() as client:
            search_resp = await client.get(
                "https://www.ultimate-guitar.com/search.php",
                params={"search_type": "title", "value": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                },
                timeout=60,
                follow_redirects=True,
            )

        if search_resp.status_code != 200:
            print(f"UG SEARCH: {search_resp.status_code}")
            return None

        data_match = re.search(r'data-content="([^"]+)"', search_resp.text)
        if not data_match:
            print("UG SEARCH: no data-content encontrado")
            return None

        raw_json = html_module.unescape(data_match.group(1))
        data = _json.loads(raw_json)
        results = data.get("store", {}).get("page", {}).get("data", {}).get("results", [])

        chord_tab = next(
            (r for r in results if r.get("type") == "Chords" and r.get("tab_url")),
            None
        )

        if not chord_tab:
            print("UG SEARCH: sin tabs de acordes en resultados")
            return None

        tab_url = chord_tab["tab_url"]
        print(f"UG: tab encontrada en {tab_url}")

        async with httpx.AsyncClient() as client:
            tab_resp = await client.get(
                tab_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=60,
                follow_redirects=True,
            )

        if tab_resp.status_code != 200:
            return None

        tab_data_match = re.search(r'data-content="([^"]+)"', tab_resp.text)
        if not tab_data_match:
            return None

        tab_raw = html_module.unescape(tab_data_match.group(1))
        tab_data = _json.loads(tab_raw)
        tab_content = (
            tab_data.get("store", {})
            .get("page", {})
            .get("data", {})
            .get("tab_view", {})
            .get("wiki_tab", {})
            .get("content", "")
        )

        if not tab_content:
            return None

        chord_pattern = re.compile(r'\[ch\]([A-G][#b]?(?:m|maj|min|sus|dim|aug|add)?[0-9]?)\[/ch\]')
        raw_chords = chord_pattern.findall(tab_content)

        if len(raw_chords) < 4:
            return None

        simplified = [simplify_chord_name(c) for c in raw_chords]
        valid = [c for c in simplified if c in KNOWN_CHORDS]

        if len(valid) < 4:
            return None

        print(f"UG OK: {len(valid)} acordes para '{artist} - {title}'")
        return valid

    except Exception as e:
        print(f"UG EXCEPTION: {e}")
        return None


def chords_to_progression(chords: list[str]) -> list[str]:
    deduped = []
    for c in chords:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    return list(dict.fromkeys(deduped))[:8]


async def assign_chords_with_groq(
    artist: str,
    title: str,
    key_en: str,
    synced_lyrics: list[dict],
    real_chords: list[str] | None = None,
) -> dict | None:
    """
    Groq asigna un acorde a cada línea de la letra real de LRCLIB.
    Si tenemos acordes reales de Cifra Club / UG, se los pasamos como contexto.
    """
    import httpx
    import json
    import re

    GROQ_KEY = "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"
    lyrics_lines = [item["text"] for item in synced_lyrics]
    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lyrics_lines))

    chords_context = ""
    if real_chords:
        prog = chords_to_progression(real_chords)
        chords_context = f"\nACORDES REALES de la canción: {' - '.join(prog)}\nUsá ESTOS acordes. No inventes otros.\n"

    prompt = f"""Sos un guitarrista experto.
La canción es "{title}" de {artist}, tonalidad {key_en if key_en else "original"}.
{chords_context}
Letra EXACTA de la canción:
{numbered}

Asigná UN acorde a CADA línea.
- Solo array JSON. Sin texto. Sin markdown.
- Exactamente {len(lyrics_lines)} objetos: [{{"line": 1, "chord": "Am"}}, ...]
- Solo: C, D, E, F, G, A, B, Am, Bm, Cm, Dm, Em, Fm, Gm
- Si tenés acordes reales arriba, respetálos y distribuílos
- Coros repiten la misma progresión"""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )

        if resp.status_code != 200:
            print(f"GROQ ERROR: {resp.status_code}")
            return None

        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(text)

        if not isinstance(parsed, list) or len(parsed) < 4:
            return None

        chord_map = {}
        for item in parsed:
            if isinstance(item, dict):
                line_num = item.get("line")
                chord = simplify_chord_name(str(item.get("chord", "")).strip())
                if line_num and chord in KNOWN_CHORDS:
                    chord_map[int(line_num)] = chord

        if len(chord_map) < 4:
            return None

        lyric_chords = []
        for i, sl in enumerate(synced_lyrics):
            chord = chord_map.get(i + 1) or (lyric_chords[-1]["chord"] if lyric_chords else get_fallback_chord(key_en, i))
            lyric_chords.append({"lyric": sl["text"], "chord": chord, "time_seconds": sl["time_seconds"]})

        all_chords = [lc["chord"] for lc in lyric_chords]
        deduped = []
        for c in all_chords:
            if not deduped or c != deduped[-1]:
                deduped.append(c)
        main_progression = list(dict.fromkeys(deduped))[:8]

        source = "cifraclub+groq" if real_chords else "groq-lrclib"
        print(f"GROQ ASSIGN OK: {len(lyric_chords)} líneas, fuente={source}, progresión={main_progression}")

        return {
            "lyric_chords": lyric_chords,
            "chords": deduped[:20],
            "main_progression": main_progression,
            "source": source,
            "capo": None,
        }

    except Exception as e:
        print(f"GROQ ASSIGN EXCEPTION: {e}")
        return None


async def fetch_chords_from_web(artist: str, title: str, key_en: str = "", synced_lyrics: list[dict] | None = None) -> dict | None:
    """
    Estrategia C — acordes reales de múltiples fuentes:
    1. Cifra Club (rock/pop latino)
    2. Ultimate Guitar (inglés y catálogo global)
    3. Groq mapea acordes reales sobre la letra de LRCLIB
    4. Fallback: Groq solo con su conocimiento
    """
    import httpx
    import json
    import re

    if not artist and not title:
        return None

    GROQ_KEY = "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"

    # ── Paso 1: acordes reales de fuentes externas ────────────────────────────
    real_chords: list[str] | None = None

    cifra_chords = await search_cifraclub(artist, title)
    if cifra_chords:
        real_chords = cifra_chords
        print(f"FUENTE: Cifra Club ({len(real_chords)} acordes)")
    else:
        ug_chords = await search_ultimate_guitar(artist, title)
        if ug_chords:
            real_chords = ug_chords
            print(f"FUENTE: Ultimate Guitar ({len(real_chords)} acordes)")
        else:
            print("FUENTE: ninguna fuente externa → solo Groq")

    # ── Paso 2: si tenemos letra de LRCLIB, Groq mapea acordes a cada línea ──
    if synced_lyrics and len(synced_lyrics) >= 4:
        result = await assign_chords_with_groq(artist, title, key_en, synced_lyrics, real_chords)
        if result:
            return result

    # ── Paso 3: si solo tenemos acordes reales pero no letra ─────────────────
    if real_chords:
        deduped = chords_to_progression(real_chords)
        return {
            "lyric_chords": [],
            "chords": real_chords[:20],
            "main_progression": deduped,
            "source": "cifraclub-nolrc",
            "capo": None,
        }

    # ── Paso 4: fallback completo — Groq genera letra+acordes ────────────────
    print("GROQ: fallback completo (sin fuentes externas ni letra LRCLIB)")
    prompt_fallback = f"""Sos un guitarrista experto. Escribí la letra de "{title}" de {artist} con el acorde de cada línea.
Tonalidad: {key_en if key_en else "original de la canción"}.
Respondé SOLO con array JSON. Sin texto, sin markdown.
Formato: [{{"lyric": "...", "chord": "Am"}}, ...]
Solo acordes: C, D, E, F, G, A, B, Am, Bm, Cm, Dm, Em, Fm, Gm
Si no conocés la canción respondé: []"""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 3000,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt_fallback}],
                },
                timeout=60,
            )

        if resp.status_code != 200:
            return None

        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(text)

        if not isinstance(parsed, list) or len(parsed) < 2:
            return None

        lyric_chords = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            lyric = item.get("lyric", "").strip()
            chord = simplify_chord_name(item.get("chord", "").strip())
            if lyric and chord in KNOWN_CHORDS:
                lyric_chords.append({"lyric": lyric, "chord": chord})

        if len(lyric_chords) < 4:
            return None

        deduped = chords_to_progression([lc["chord"] for lc in lyric_chords])
        return {
            "lyric_chords": lyric_chords,
            "chords": [lc["chord"] for lc in lyric_chords][:20],
            "main_progression": deduped,
            "source": "groq-llama",
            "capo": None,
        }

    except Exception as e:
        print(f"GROQ FALLBACK EXCEPTION: {e}")
        return None


def get_fallback_chord(key_en: str, position: int) -> str:
    """Progresiones comunes según tonalidad para cuando Groq no responde."""
    progressions = {
        "C major":  ["C", "Am", "F", "G"],
        "A minor":  ["Am", "F", "C", "G"],
        "G major":  ["G", "Em", "C", "D"],
        "E minor":  ["Em", "C", "G", "D"],
        "D major":  ["D", "Bm", "G", "A"],
        "A major":  ["A", "F", "D", "E"],
        "E major":  ["E", "C", "A", "B"],
        "F major":  ["F", "Dm", "C", "G"],
        "D minor":  ["Dm", "C", "F", "G"],
        "G minor":  ["Gm", "F", "Cm", "D"],
    }
    chords = progressions.get(key_en, ["Am", "F", "C", "G"])
    return chords[position % len(chords)]


def build_timeline_from_lyric_chords(lyric_chords: list[dict], synced_lyrics: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Construye la timeline combinando acordes y timestamps.

    Modo LRCLIB (preferido): si lyric_chords ya tiene "time_seconds" (asignados
    directamente por Groq sobre la letra real), los usa directamente → timing perfecto.

    Modo fallback: si no hay "time_seconds", usa similitud de texto para mapear
    los lyric_chords de Groq contra synced_lyrics de LRCLIB.
    """
    if not lyric_chords or not synced_lyrics:
        return [], []

    timeline = []
    seen_last = None

    # ── Modo directo: Groq trabajó sobre la letra real de LRCLIB ─────────────
    if "time_seconds" in lyric_chords[0]:
        for lc in lyric_chords:
            chord = lc["chord"]
            ts = lc["time_seconds"]
            minutes = int(ts // 60)
            secs = int(ts % 60)
            if chord != seen_last:
                timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
                seen_last = chord

        all_chords = [lc["chord"] for lc in lyric_chords]
        deduped = []
        for c in all_chords:
            if not deduped or c != deduped[-1]:
                deduped.append(c)
        progression = list(dict.fromkeys(deduped))[:8]
        return timeline[:30], progression

    # ── Modo similitud: fallback cuando Groq generó su propia letra ──────────
    import difflib

    def normalize(s: str) -> str:
        return s.lower().strip().replace("'", "").replace(",", "").replace(".", "")

    synced_texts = [normalize(l["text"]) for l in synced_lyrics]
    lyric_idx = 0

    for lc in lyric_chords:
        groq_text = normalize(lc["lyric"])
        chord = lc["chord"]

        best_match_idx = lyric_idx
        best_ratio = 0.0
        search_start = max(0, lyric_idx - 2)
        search_end = min(len(synced_lyrics), lyric_idx + 15)

        for i in range(search_start, search_end):
            ratio = difflib.SequenceMatcher(None, groq_text, synced_texts[i]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_idx = i

        if best_ratio > 0.3:
            lyric_idx = best_match_idx + 1
            ts = synced_lyrics[best_match_idx]["time_seconds"]
        else:
            if lyric_idx < len(synced_lyrics):
                ts = synced_lyrics[lyric_idx]["time_seconds"]
                lyric_idx += 1
            else:
                continue

        minutes = int(ts // 60)
        secs = int(ts % 60)
        if chord != seen_last:
            timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
            seen_last = chord

    all_chords = [lc["chord"] for lc in lyric_chords]
    deduped = []
    for c in all_chords:
        if not deduped or c != deduped[-1]:
            deduped.append(c)
    progression = list(dict.fromkeys(deduped))[:8]

    return timeline[:20], progression


def build_timeline_from_lyrics(sections: list[dict], synced_lyrics: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Mapea los acordes de cada sección sobre los timestamps reales de la letra.
    Usa lyric_lines para saber cuántas líneas corresponden a cada sección.
    """
    if not sections or not synced_lyrics:
        return [], []

    timeline = []
    seen_last = None
    lyric_idx = 0
    total_lyrics = len(synced_lyrics)

    for section in sections:
        chords = [c for c in section.get("chords", []) if c in KNOWN_CHORDS]
        repetitions = section.get("repetitions", 1)
        lyric_lines = section.get("lyric_lines", 4)
        if not chords:
            continue

        for rep in range(repetitions):
            lines_this_rep = lyric_lines
            if lyric_idx >= total_lyrics:
                break

            # Distribuir los acordes de la sección en las líneas disponibles
            lines_per_chord = max(1, lines_this_rep // len(chords))

            for chord_idx, chord in enumerate(chords):
                if lyric_idx >= total_lyrics:
                    break
                ts = synced_lyrics[lyric_idx]["time_seconds"]
                minutes = int(ts // 60)
                secs = int(ts % 60)
                if chord != seen_last:
                    timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
                    seen_last = chord
                # Avanzar líneas de letra
                lyric_idx = min(lyric_idx + lines_per_chord, total_lyrics)

    # Progresión única
    all_chords = []
    for s in sections:
        all_chords.extend([c for c in s.get("chords", []) if c in KNOWN_CHORDS])
    progression = list(dict.fromkeys(all_chords))[:8]

    return timeline[:20], progression


def build_timeline_from_chords(chords: list[str], total_duration: float, sections: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    """
    Construye timeline a partir de acordes reales.
    Si hay secciones (de Groq), las usa para distribuir el timing proporcionalmente.
    Si no, distribuye uniformemente.
    """
    if not chords:
        return [], []

    if sections:
        # Calcular total de "slots" de acorde pesados por repeticiones
        total_slots = sum(
            len(s.get("chords", [])) * s.get("repetitions", 1)
            for s in sections
        )
        if total_slots == 0:
            total_slots = len(chords)

        slot_duration = total_duration / total_slots
        timeline = []
        current_time = 0.0
        seen_last = None

        for section in sections:
            section_chords = [c for c in section.get("chords", []) if c in KNOWN_CHORDS]
            repetitions = section.get("repetitions", 1)
            if not section_chords:
                continue
            for _ in range(repetitions):
                for chord in section_chords:
                    minutes = int(current_time // 60)
                    secs = int(current_time % 60)
                    if chord != seen_last:
                        timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
                        seen_last = chord
                    current_time += slot_duration
    else:
        # Distribución uniforme
        segment_duration = total_duration / len(chords)
        timeline = []
        seen_last = None
        for i, chord in enumerate(chords):
            seconds = i * segment_duration
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            if chord != seen_last:
                timeline.append({"time": f"{minutes}:{secs:02d}", "chord": chord})
                seen_last = chord

    # Progresión: acordes únicos sin repetidos, máx 8
    progression = []
    for c in chords:
        if not progression or c != progression[-1]:
            progression.append(c)
    progression = list(dict.fromkeys(progression))[:8]

    return timeline[:20], progression


# ─── Identificación de canción con AcoustID ──────────────────────────────────

ACOUSTID_API_KEY = "cSpUJKpD"  # Key pública de prueba de AcoustID

def identify_song(audio_path: str) -> dict:
    """
    Usa fpcalc para generar el fingerprint y AcoustID + MusicBrainz
    para identificar artista y título reales de la canción.
    Devuelve {"title": str, "artist": str} o {} si no se pudo identificar.
    """
    import subprocess
    import json
    import urllib.request
    import urllib.parse

    # Buscar fpcalc en la carpeta del script o en PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fpcalc_path = os.path.join(script_dir, "fpcalc.exe")
    if not os.path.exists(fpcalc_path):
        fpcalc_path = "fpcalc"  # Intentar desde PATH

    try:
        result = subprocess.run(
            [fpcalc_path, "-json", audio_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {}
        fp_data = json.loads(result.stdout)
        fingerprint = fp_data.get("fingerprint", "")
        duration = fp_data.get("duration", 0)
    except Exception:
        return {}

    # Consultar AcoustID
    params = urllib.parse.urlencode({
        "client": ACOUSTID_API_KEY,
        "duration": int(duration),
        "fingerprint": fingerprint,
        "meta": "recordings",
    })
    url = f"https://api.acoustid.org/v2/lookup?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return {}

    if data.get("status") != "ok":
        return {}

    results = data.get("results", [])
    if not results:
        return {}

    # Tomar el resultado con mayor score
    best = max(results, key=lambda r: r.get("score", 0))
    recordings = best.get("recordings", [])
    if not recordings:
        return {}

    rec = recordings[0]
    title = rec.get("title", "")
    artists = rec.get("artists", [])
    artist = artists[0].get("name", "") if artists else ""

    return {"title": title, "artist": artist}


# ─── Endpoint principal ───────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Solo aceptamos archivos .mp3")

    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Cargar el audio con librosa
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=120)

        # ── BPM ──────────────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(np.atleast_1d(tempo)[0])))

        # ── Chroma global para tonalidad ──────────────────────────────────────
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_en, key_es = detect_key(chroma_mean)

        # ── Notas de la tonalidad ─────────────────────────────────────────────
        notes = get_notes_from_key(key_en)

        # ── Capo ──────────────────────────────────────────────────────────────
        capo = None

        # ── Identificación real de la canción con AcoustID ────────────────────
        song_info = identify_song(tmp_path)
        if song_info.get("title"):
            estimated_title = song_info["title"]
            artist = song_info.get("artist", "")
            display_title = f"{artist} - {estimated_title}" if artist else estimated_title
        else:
            estimated_title = file.filename.replace(".mp3", "").replace("-", " ").replace("_", " ").title()
            display_title = estimated_title
            artist = ""

        # ── Letra sincronizada de LRCLIB (se busca ANTES de Groq) ────────────
        synced_lyrics = await fetch_lyrics_from_lrclib(artist, estimated_title)
        if synced_lyrics:
            print(f"LRCLIB OK: {len(synced_lyrics)} líneas sincronizadas para '{artist} - {estimated_title}'")
        else:
            print(f"LRCLIB: no encontró letra para '{artist} - {estimated_title}'")

        # ── Decisión MVP: curada verificada o Groq ───────────────────────────
        curated_song = find_curated_song(artist, estimated_title)
        has_exact_timing = False
        chord_timeline_out: list = []

        if (
            curated_song
            and curated_song.get("timing_status") == "verified"
            and curated_song.get("chord_timeline")
        ):
            # Caso A: canción curada con timing verificado → usar chord_timeline exacto
            ct = curated_song["chord_timeline"]
            print(f"BASE CURADA VERIFICADA: usando chord_timeline exacto ({len(ct)} entradas)")
            timeline, chord_timeline_out, simplified_progression = build_from_chord_timeline(ct)
            chord_source = "curated"
            has_exact_timing = True
            capo = curated_song.get("capo") or capo
        else:
            # Caso B: canción no verificada o no curada → Groq como motor principal
            if curated_song:
                print(f"BASE CURADA: encontrada pero sin timing verificado → usando Groq")
            external_chords = await fetch_chords_from_web(artist, estimated_title, key_en, synced_lyrics)

        if not has_exact_timing:
            if external_chords:
                duration = len(y) / sr

                if external_chords.get("lyric_chords"):
                    timeline, simplified_progression = build_timeline_from_lyric_chords(
                        external_chords["lyric_chords"], synced_lyrics
                    )
                elif external_chords.get("sections"):
                    timeline, simplified_progression = build_timeline_from_lyrics(
                        external_chords["sections"], synced_lyrics
                    )
                else:
                    timeline, simplified_progression = build_timeline_from_chords(
                        external_chords["chords"], duration, external_chords.get("sections")
                    )

                chord_source = external_chords["source"]
                if external_chords.get("capo") is not None:
                    capo = external_chords["capo"]
            else:
                external_chords = {}
                timeline, simplified_progression = build_timeline_from_key(y, sr, key_en)
                if not timeline:
                    segment_seconds = 8.0
                    hop_samples = int(segment_seconds * sr)
                    segments = [y[i:i + hop_samples] for i in range(0, len(y), hop_samples) if len(y[i:i + hop_samples]) > sr]
                    chords_per_segment = []
                    for seg in segments:
                        seg_chroma = librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1)
                        chords_per_segment.append(chroma_to_chord(seg_chroma))
                    simplified_progression = simplify_progression(chords_per_segment)
                    timeline = build_timeline(chords_per_segment, hop_samples, sr, segment_seconds)
                chord_source = "key-based"

        return {
            "fileName": file.filename,
            "estimatedTitle": display_title,
            "searchTitle": f"{artist} {estimated_title}".strip() if song_info else estimated_title,
            "chordSource": chord_source,
            "curatedId": curated_song.get("id") if curated_song else None,
            "hasExactTiming": has_exact_timing,
            "chordTimeline": chord_timeline_out,
            "confidence": external_chords.get("confidence", 0) if not has_exact_timing else 1,
            "sections": external_chords.get("sections", []) if not has_exact_timing else [],
            "mainProgression": external_chords.get("main_progression", simplified_progression) if not has_exact_timing else simplified_progression,
            "warning": external_chords.get("warning") if not has_exact_timing else None,
            "key": key_es,
            "bpm": bpm,
            "simplifiedProgression": simplified_progression,
            "timeline": timeline,
            "tuning": "E A D G B E",
            "capo": capo,
            "notes": notes,
            "summary": (
                f"Tonalidad detectada: {key_es} — {bpm} BPM. Versión curada con acordes reales para tocar."
                if chord_source == "curated"
                else f"Tonalidad detectada: {key_es} — {bpm} BPM. Progresión simplificada para principiantes."
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar el audio: {str(e)}")

    finally:
        os.unlink(tmp_path)


@app.get("/lyrics")
async def get_lyrics(title: str):
    """Busca letra sincronizada en LRCLIB (gratis, sin token, con timestamps)."""
    import httpx
    import re

    def parse_lrc(lrc_text: str) -> list[dict]:
        """Parsea formato LRC y devuelve lista de {time_seconds, text}."""
        lines = []
        pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
        for line in lrc_text.splitlines():
            match = pattern.match(line.strip())
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                text = match.group(3).strip()
                if text:  # ignorar líneas vacías
                    lines.append({
                        "time_seconds": minutes * 60 + seconds,
                        "text": text
                    })
        return lines

    # Limpiar el título para mejor búsqueda
    clean_title = title.strip()

    async with httpx.AsyncClient() as client:
        # Buscar en LRCLIB
        resp = await client.get(
            "https://lrclib.net/api/search",
            params={"q": clean_title},
            headers={"User-Agent": "EasyChordsIA/1.0"},
            timeout=60,
        )

    if resp.status_code != 200:
        return {"lyrics": [], "syncedLyrics": [], "found": False, "title": title}

    results = resp.json()
    if not results:
        return {"lyrics": [], "syncedLyrics": [], "found": False, "title": title}

    # Preferir el primer resultado con letra sincronizada
    chosen = None
    for r in results:
        if r.get("syncedLyrics"):
            chosen = r
            break
    # Si no hay sincronizada, usar la primera con letra plana
    if not chosen:
        for r in results:
            if r.get("plainLyrics"):
                chosen = r
                break

    if not chosen:
        return {"lyrics": [], "syncedLyrics": [], "found": False, "title": title}

    song_title = f"{chosen.get('artistName', '')} - {chosen.get('trackName', '')}".strip(" -")

    # Letra sincronizada (con timestamps)
    synced_raw = chosen.get("syncedLyrics", "")
    synced = parse_lrc(synced_raw) if synced_raw else []

    # Letra plana como fallback
    plain = [l for l in (chosen.get("plainLyrics") or "").splitlines() if l.strip()]

    return {
        "lyrics": plain,
        "syncedLyrics": synced,   # [{time_seconds: float, text: str}, ...]
        "found": True,
        "title": song_title,
    }


@app.get("/chords")
async def get_chords(artist: str, title: str):
    """Debug: busca acordes en fuentes externas."""
    result = await fetch_chords_from_web(artist, title)
    return result or {"found": False, "chords": [], "source": "none"}


@app.get("/")
def root():
    return {"status": "Easy Chords IA backend corriendo", "version": "1.0"}

@app.post("/save-chord-timeline")
async def save_chord_timeline(payload: dict):
    """
    Recibe { curated_id, chord_timeline } y persiste el chord_timeline
    en la canción correspondiente dentro de canciones_curadas.json.
    También marca timing_status = "verified".
    """
    import os, json as _json

    curated_id = payload.get("curated_id", "").strip()
    chord_timeline = payload.get("chord_timeline", [])

    if not curated_id:
        raise HTTPException(status_code=400, detail="curated_id es requerido")
    if not isinstance(chord_timeline, list) or len(chord_timeline) == 0:
        raise HTTPException(status_code=400, detail="chord_timeline debe ser un array no vacío")

    # Validar estructura de cada entrada
    for entry in chord_timeline:
        if "time_seconds" not in entry or "chord" not in entry:
            raise HTTPException(status_code=400, detail="Cada entrada debe tener time_seconds y chord")

    candidates = [
        os.path.join(os.path.dirname(__file__), "canciones_curadas.json"),
        os.path.join(os.path.dirname(__file__), "..", "App", "canciones_curadas.json"),
    ]

    db_path = None
    for path in candidates:
        if os.path.exists(path):
            db_path = path
            break

    if not db_path:
        raise HTTPException(status_code=404, detail="canciones_curadas.json no encontrado")

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = _json.load(f)

        # Buscar canción por id
        target = None
        for cancion in db.get("canciones", []):
            if cancion.get("id") == curated_id:
                target = cancion
                break

        if target is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró canción con id='{curated_id}'"
            )

        # Actualizar chord_timeline y marcar como verificado
        target["chord_timeline"] = chord_timeline
        target["timing_status"] = "verified"

        # Guardar preservando encoding y tildes
        with open(db_path, "w", encoding="utf-8") as f:
            _json.dump(db, f, ensure_ascii=False, indent=2)

        print(f"SAVE_CHORD_TIMELINE: guardado '{curated_id}' → {len(chord_timeline)} entradas")

        return {
            "ok": True,
            "curated_id": curated_id,
            "entries": len(chord_timeline),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar: {str(e)}")


@app.get("/songs")
def get_songs():
    """
    Devuelve la lista de canciones curadas disponibles en la base.
    Lee canciones_curadas.json y retorna artista, titulo e id.
    """
    import os

    candidates = [
        os.path.join(os.path.dirname(__file__), "canciones_curadas.json"),
        os.path.join(os.path.dirname(__file__), "..", "App", "canciones_curadas.json"),
    ]

    db_path = None
    for path in candidates:
        if os.path.exists(path):
            db_path = path
            break

    if not db_path:
        return {"error": "canciones_curadas.json no encontrado", "songs": []}

    try:
        import json as _json
        with open(db_path, "r", encoding="utf-8") as f:
            db = _json.load(f)

        songs = [
            {
                "id": c.get("id", ""),
                "artista": c.get("artista", ""),
                "titulo": c.get("titulo", ""),
                "genero": c.get("genero", ""),
                "tonalidad": c.get("tonalidad", ""),
                "estado": c.get("estado", ""),
            }
            for c in db.get("canciones", [])
        ]

        return {"total": len(songs), "songs": songs}

    except Exception as e:
        return {"error": f"Error leyendo base de canciones: {e}", "songs": []}

