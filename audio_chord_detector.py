"""
audio_chord_detector.py
=======================
Módulo aislado para detección automática de chord_timeline a partir de audio.

Entrada : señal de audio ya cargada por librosa (y, sr)
Salida  : lista de { time_seconds, chord, confidence }

NO depende de main.py ni de ningún otro módulo del proyecto.
Usa solo librosa y numpy, que ya están instalados en el backend.

Limitación conocida: la detección es estimada (audio complejo = guitarra + voz + batería).
Para producción, el resultado debe combinarse con la base curada si existe.
"""

from __future__ import annotations

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NOTE_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B"]

# Duración de cada ventana de análisis en segundos.
# 0.75 s es un buen balance entre resolución temporal y estabilidad del chroma.
WINDOW_SECONDS: float = 0.75

# Un acorde debe durar al menos este tiempo para no ser considerado ruido.
MIN_CHORD_DURATION: float = 0.5

# Energía RMS mínima para considerar que hay contenido musical en la ventana.
# Ventanas por debajo de este umbral se marcan como "silencio" y se ignoran.
SILENCE_THRESHOLD: float = 0.01

# Número de vecinos a considerar al suavizar la secuencia de acordes.
# Reduce flickers: si una ventana aislada difiere de sus vecinas, se la reemplaza.
SMOOTHING_WINDOW: int = 3


# ---------------------------------------------------------------------------
# Plantillas de acordes para matching por correlación de chroma
# ---------------------------------------------------------------------------

def _build_chord_templates() -> dict[str, np.ndarray]:
    """
    Construye un diccionario de plantillas chroma para acordes mayores,
    menores, séptimas dominantes y menores con séptima.

    Cada plantilla es un vector de 12 bins (uno por semitono).
    La correlación entre el chroma de un frame y estas plantillas
    determina el acorde más probable.
    """
    templates: dict[str, np.ndarray] = {}

    for i, note in enumerate(NOTE_NAMES):
        root = i % 12

        # Mayor: raíz + tercera mayor (4 st) + quinta (7 st)
        major = np.zeros(12)
        major[root] = 1.0
        major[(root + 4) % 12] = 1.0
        major[(root + 7) % 12] = 1.0
        templates[note] = major

        # Menor: raíz + tercera menor (3 st) + quinta (7 st)
        minor = np.zeros(12)
        minor[root] = 1.0
        minor[(root + 3) % 12] = 1.0
        minor[(root + 7) % 12] = 1.0
        templates[f"{note}m"] = minor

        # Séptima dominante: mayor + séptima menor (10 st)
        dom7 = major.copy()
        dom7[(root + 10) % 12] = 0.8  # peso ligeramente menor
        templates[f"{note}7"] = dom7

        # Menor séptima: menor + séptima menor (10 st)
        min7 = minor.copy()
        min7[(root + 10) % 12] = 0.8
        templates[f"{note}m7"] = min7

    return templates


# Construido una sola vez al importar el módulo.
CHORD_TEMPLATES: dict[str, np.ndarray] = _build_chord_templates()


# ---------------------------------------------------------------------------
# Mapeo de enharmonics / simplificación
# ---------------------------------------------------------------------------

# Si el acorde detectado tiene un alias más común en guitarra, lo normalizamos.
ENHARMONIC_MAP: dict[str, str] = {
    "C#":  "C#", "Db":  "C#",
    "D#":  "Eb", "Eb":  "Eb",
    "F#":  "F#", "Gb":  "F#",
    "G#":  "G#", "Ab":  "G#",
    "A#":  "A#", "Bb":  "A#",
    "C#m": "C#m", "Dbm": "C#m",
    "D#m": "Ebm", "Ebm": "Ebm",
    "F#m": "F#m", "Gbm": "F#m",
    "G#m": "G#m", "Abm": "G#m",
    "A#m": "Bbm", "Bbm": "Bbm",
}


def _normalize_chord(chord: str) -> str:
    """Normaliza enharmonics y devuelve el nombre canónico del acorde."""
    return ENHARMONIC_MAP.get(chord, chord)


# ---------------------------------------------------------------------------
# Función core: detectar el acorde más probable en un vector chroma
# ---------------------------------------------------------------------------

def _chroma_to_chord_scored(
    chroma: np.ndarray,
    allowed_chords: set[str] | None = None,
) -> tuple[str, float]:
    """
    Dado un vector chroma de 12 bins, devuelve (chord_name, confidence).

    Si allowed_chords está definido, se evalúan primero los acordes permitidos.
    Si ninguno supera el umbral mínimo de confianza, se cae al mejor global.

    confidence es el cosine similarity entre el chroma y la plantilla ganadora.
    """
    norm = np.linalg.norm(chroma)
    if norm < 1e-6:
        # Vector casi nulo = silencio
        return ("N/A", 0.0)

    chroma_norm = chroma / norm

    best_chord = "C"
    best_score = -1.0
    best_allowed = None
    best_allowed_score = -1.0

    for chord_name, template in CHORD_TEMPLATES.items():
        score = float(np.dot(chroma_norm, template / np.linalg.norm(template)))
        normalized_name = _normalize_chord(chord_name)

        if score > best_score:
            best_score = score
            best_chord = normalized_name

        # Registrar también el mejor dentro de allowed_chords
        if allowed_chords and normalized_name in allowed_chords:
            if score > best_allowed_score:
                best_allowed_score = score
                best_allowed = normalized_name

    # Si hay un acorde permitido razonablemente cercano al mejor global,
    # preferirlo. "Razonablemente cercano" = dentro del 15 % del score máximo.
    if best_allowed is not None and best_allowed_score >= best_score * 0.85:
        return (best_allowed, best_allowed_score)

    return (_normalize_chord(best_chord), best_score)


# ---------------------------------------------------------------------------
# Suavizado de secuencia: eliminar flickers
# ---------------------------------------------------------------------------

def _smooth_chord_sequence(
    chords: list[str],
    window: int = SMOOTHING_WINDOW,
) -> list[str]:
    """
    Reemplaza acordes aislados (rodeados de acordes distintos iguales entre sí)
    por el acorde mayoritario en la vecindad.

    Ejemplo: [Em, Em, B7, Em, Em] → [Em, Em, Em, Em, Em]
    """
    if len(chords) <= window:
        return chords

    smoothed = chords[:]
    half = window // 2

    for i in range(half, len(chords) - half):
        neighborhood = chords[i - half: i + half + 1]
        # Contar frecuencias en la vecindad
        counts: dict[str, int] = {}
        for c in neighborhood:
            counts[c] = counts.get(c, 0) + 1
        majority = max(counts, key=lambda k: counts[k])
        smoothed[i] = majority

    return smoothed


# ---------------------------------------------------------------------------
# Agrupación: colapsar entradas consecutivas con el mismo acorde
# ---------------------------------------------------------------------------

def _group_consecutive(
    timed_chords: list[tuple[float, str, float]],
    min_duration: float = MIN_CHORD_DURATION,
) -> list[dict]:
    """
    Recibe lista de (time_seconds, chord, confidence) y colapsa entradas
    consecutivas con el mismo acorde en una sola entrada.

    También descarta acordes que duran menos de min_duration segundos
    (probable ruido o artefacto de detección).

    Devuelve lista de { time_seconds, chord, confidence }.
    """
    if not timed_chords:
        return []

    grouped: list[dict] = []
    current_time, current_chord, current_conf = timed_chords[0]
    conf_accumulator = [current_conf]

    for i in range(1, len(timed_chords)):
        t, chord, conf = timed_chords[i]

        if chord == current_chord:
            # Mismo acorde: acumular confianza para promediar
            conf_accumulator.append(conf)
        else:
            # Nuevo acorde: calcular duración del anterior
            duration = t - current_time
            if duration >= min_duration and current_chord != "N/A":
                grouped.append({
                    "time_seconds": round(current_time, 2),
                    "chord": current_chord,
                    "confidence": round(float(np.mean(conf_accumulator)), 3),
                })
            # Iniciar nuevo acorde
            current_time = t
            current_chord = chord
            conf_accumulator = [conf]

    # Agregar el último grupo (sin saber su duración exacta, incluir siempre)
    if current_chord != "N/A":
        grouped.append({
            "time_seconds": round(current_time, 2),
            "chord": current_chord,
            "confidence": round(float(np.mean(conf_accumulator)), 3),
        })

    return grouped


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def detect_chord_timeline(
    y: np.ndarray,
    sr: int,
    allowed_chords: list[str] | None = None,
) -> list[dict]:
    """
    Detecta un chord_timeline estimado a partir de una señal de audio.

    Parámetros
    ----------
    y : np.ndarray
        Señal de audio mono (ya cargada con librosa.load).
    sr : int
        Sample rate de la señal.
    allowed_chords : list[str] | None
        Si se pasa, los acordes detectados se restringen a esta lista
        cuando la confianza lo permite (tipicamente los acordes de la
        cancion curada). Si es None, se detectan todos los acordes posibles.

    Retorna
    -------
    list[dict]
        Lista de entradas { "time_seconds": float, "chord": str, "confidence": float }
        ordenadas por tiempo. Acordes consecutivos iguales ya están colapsados.
        Lista vacía si el audio es demasiado corto o silencioso.

    Notas
    -----
    - Esto es detección estimada, no perfecta. El audio mezclado (voz + batería
      + bajo + guitarra) introduce ambigüedad en el chroma.
    - allowed_chords mejora la precisión cuando se conoce la tonalidad/acordes
      reales de la canción.
    - El resultado debe considerarse como punto de partida, no como verdad absoluta.
    """

    # Asegurar mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Tamaño de ventana en samples
    window_samples = int(WINDOW_SECONDS * sr)

    if len(y) < window_samples:
        print("CHORD_DETECTOR: audio demasiado corto, no se puede detectar timeline")
        return []

    allowed_set: set[str] | None = set(allowed_chords) if allowed_chords else None

    # ── Paso 1: calcular chroma por ventana ──────────────────────────────────
    # Usamos chroma_cqt porque es más robusto ante variaciones de timbre
    # (voz, percusión) que chroma_stft.
    hop_samples = window_samples // 2  # 50 % de overlap para mejor resolución
    chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_samples)
    # chroma_full: shape (12, n_frames)

    # Energía RMS por frame para detectar silencio
    rms = librosa.feature.rms(y=y, frame_length=window_samples, hop_length=hop_samples)[0]

    n_frames = chroma_full.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_samples)

    print(f"CHORD_DETECTOR: analizando {n_frames} frames "
          f"({WINDOW_SECONDS}s ventana, {len(y)/sr:.1f}s total)")

    # ── Paso 2: detectar acorde por frame ────────────────────────────────────
    raw_sequence: list[tuple[float, str, float]] = []  # (time, chord, confidence)

    for i in range(n_frames):
        frame_chroma = chroma_full[:, i]
        energy = float(rms[i]) if i < len(rms) else 0.0

        # Ignorar frames silenciosos
        if energy < SILENCE_THRESHOLD:
            continue

        chord, confidence = _chroma_to_chord_scored(frame_chroma, allowed_set)
        raw_sequence.append((float(frame_times[i]), chord, confidence))

    if not raw_sequence:
        print("CHORD_DETECTOR: no se detectaron frames con energía suficiente")
        return []

    print(f"CHORD_DETECTOR: {len(raw_sequence)} frames con contenido musical")

    # ── Paso 3: suavizar la secuencia para eliminar flickers ─────────────────
    chords_only = [c for _, c, _ in raw_sequence]
    smoothed_chords = _smooth_chord_sequence(chords_only, window=SMOOTHING_WINDOW)

    smoothed_sequence = [
        (t, smoothed_chords[i], conf)
        for i, (t, _, conf) in enumerate(raw_sequence)
    ]

    # ── Paso 4: colapsar consecutivos y filtrar ruido ─────────────────────────
    timeline = _group_consecutive(smoothed_sequence, min_duration=MIN_CHORD_DURATION)

    print(f"CHORD_DETECTOR: timeline final → {len(timeline)} cambios de acorde")
    for entry in timeline:
        print(f"  {entry['time_seconds']:6.2f}s  {entry['chord']:<6}  "
              f"conf={entry['confidence']:.2f}")

    return timeline
