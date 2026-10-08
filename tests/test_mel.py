import numpy as np
import pytest
import torch

from test_tts.audio.mel import WhisperMel

SR, HOP, N_FFT, N_MELS = 24000, 256, 1024, 96


def hf_mel(wav: np.ndarray) -> np.ndarray:
    """Reference: the HF WhisperFeatureExtractor exactly as Matcha-TTS drives it for VocBulwark."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(sampling_rate=SR, n_fft=N_FFT, feature_size=N_MELS, hop_length=HOP)
    fe.n_samples = int(30.0 * SR)
    fe.chunk_length = 30.0
    return fe(wav, sampling_rate=SR, padding="longest", return_tensors="np")["input_features"][0]


def make_wav(seconds: float, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    # a few harmonics + a quiet noise floor, so the (peak - 8) clamp actually bites somewhere
    x = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 540 * t) + 1e-3 * g.standard_normal(t.shape)
    x[: len(x) // 5] *= 1e-3  # near-silent start
    return x.astype(np.float32)


@pytest.mark.parametrize("seconds", [1.3, 2.0471])
def test_matches_hf_extractor(seconds):
    wav = make_wav(seconds, seed=0)
    ours = WhisperMel(SR, N_FFT, N_MELS, HOP)(torch.from_numpy(wav)).numpy()
    ref = hf_mel(wav)
    assert ours.shape == ref.shape == (N_MELS, len(wav) // HOP)
    np.testing.assert_allclose(ours, ref, atol=2e-4, rtol=1e-4)


def test_batched_with_lengths_matches_single():
    mel_fn = WhisperMel(SR, N_FFT, N_MELS, HOP)
    wavs = [make_wav(1.0, 1), make_wav(1.7, 2)]
    lengths = torch.tensor([len(w) for w in wavs])
    batch = torch.zeros(2, int(lengths.max()))
    for i, w in enumerate(wavs):
        batch[i, : len(w)] = torch.from_numpy(w)
    out = mel_fn(batch, lengths)
    for i, w in enumerate(wavs):
        single = mel_fn(torch.from_numpy(w))
        n = single.shape[-1]
        torch.testing.assert_close(out[i, :, :n], single, atol=1e-4, rtol=1e-4)
        assert torch.all(out[i, :, n:] == 0)
