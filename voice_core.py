"""Core logic: speech/noise detection, voiceprints, speaker matching."""
import io
import pathlib
import pickle

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
CHUNK_SEC = 0.5
MAX_EMBED_SEC = 30
DB_PATH = pathlib.Path("voiceprints.pkl")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_ecapa"
        )
    return _model


# ---------------------------------------------------------------- audio
def read_audio(file_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes (from the browser recorder) to mono float32 at 16 kHz."""
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, SR).numpy()
    return data.astype(np.float32)


def detect_speech(wav: np.ndarray, margin_db: float, abs_db: float):
    """Noise vs speech detection on 0.5 s chunks.

    Returns (levels_db, speech_mask, threshold_db, noise_floor_db).
    A chunk is speech if it is louder than the absolute minimum AND, when the
    clip contains clear pauses, louder than the noise floor + margin.
    """
    n = int(SR * CHUNK_SEC)
    count = len(wav) // n
    if count == 0:
        return np.array([]), np.array([], dtype=bool), abs_db, abs_db
    chunks = wav[: count * n].reshape(count, n)
    levels = 20 * np.log10(np.sqrt((chunks ** 2).mean(axis=1)) + 1e-10)
    floor = float(np.percentile(levels, 10))
    if levels.max() - floor >= margin_db:
        thr = max(floor + margin_db, abs_db)
    else:
        thr = abs_db
    return levels, levels > thr, float(thr), floor


def voice_noise_percent(mask: np.ndarray):
    """Return (useful voice %, noise %) based on the share of 0.5 s chunks."""
    if len(mask) == 0:
        return 0.0, 100.0
    voice = float(mask.mean() * 100)
    return voice, 100.0 - voice


def chunks_audio(wav: np.ndarray, mask: np.ndarray, start: int = 0, end=None) -> np.ndarray:
    """Concatenate the speech chunks of wav between chunk indexes start and end."""
    n = int(SR * CHUNK_SEC)
    end = len(mask) if end is None else end
    parts = [wav[i * n:(i + 1) * n] for i in range(start, end) if mask[i]]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- voiceprints
def embed(wav: np.ndarray) -> np.ndarray:
    wav = wav[: MAX_EMBED_SEC * SR]
    t = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        e = get_model().encode_batch(t).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def load_db() -> dict:
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db: dict) -> None:
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def voiceprint(embeddings: list) -> np.ndarray:
    m = np.mean(embeddings, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def score_all(e: np.ndarray, db: dict) -> dict:
    return {name: float(np.dot(e, voiceprint(embs))) for name, embs in db.items()}


def decide(scores: dict, threshold: float):
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= threshold else ("Unknown", scores[best])
