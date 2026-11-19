"""WAV decoding without torchaudio: float WAVs are read, compressed ones are refused with a reason the user sees."""
import io
import struct
import wave
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from app.ingestion.audio_engine import _decode_wav
from app.ingestion.image_engine import _read_zip

X = np.sin(np.linspace(0, 200, 16_000)).astype(np.float32) * 0.5


def _float_wav(x, rate=16_000):
    b = io.BytesIO()
    wavfile.write(b, rate, x)
    return b.getvalue()


def _pcm24(x):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(3); w.setframerate(16_000)
        v = (x * 8388607).astype(np.int32)
        w.writeframes(np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes())
    return b.getvalue()


def _adpcm():
    fmt = struct.pack("<HHIIHHHH", 0x11, 1, 8000, 4000, 256, 4, 2, 505)
    data = b"\0" * 256
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("raw", [_float_wav(X), _float_wav(X.astype(np.float64)), _pcm24(X)])
def test_float_and_24bit_wavs_decode(raw):
    np.testing.assert_allclose(_decode_wav(raw), X, atol=1e-3)


def test_float_wav_is_resampled_and_mixed_to_mono():
    stereo = np.stack([X, X], 1)
    assert len(_decode_wav(_float_wav(np.resize(stereo, (44_100, 2)), rate=44_100))) == 16_000


def test_compressed_wav_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="ADPCM"):
        _decode_wav(_adpcm())
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("dog/1.wav", _adpcm())
    with pytest.raises(ValueError, match="1.wav: compressed WAV"):
        _read_zip(b.getvalue(), "upload", [], (".wav",), _decode_wav, "clip")
