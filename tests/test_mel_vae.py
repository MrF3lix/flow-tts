import torch

from test_tts.data.mel_dataset import collate_mels
from test_tts.models.mel_vae import MelVAE


def small_vae():
    torch.manual_seed(0)
    return MelVAE(n_mels=96, latent_dim=8, compression=4, channels=[16, 24, 32], blocks_per_level=1).eval()


def test_shapes_and_losses():
    vae = small_vae()
    mel = torch.randn(2, 96, 64)
    lengths = torch.tensor([64, 50])
    out = vae(mel, lengths)
    assert out["mel_hat"].shape == mel.shape
    assert out["mu"].shape == (2, 8, 16)
    assert torch.isfinite(out["loss"]) and out["recon"] > 0 and out["kl"] >= 0
    out["loss"].backward()
    assert all(p.grad is not None for p in vae.parameters() if p.requires_grad)


def test_padding_does_not_change_valid_frames():
    """Masking is correct iff an utterance gives the same output alone and inside a padded batch."""
    vae = small_vae()
    mel = torch.randn(1, 96, 64)
    alone = vae(mel, torch.tensor([64]), sample=False)
    padded = torch.nn.functional.pad(mel, (0, 32))
    with_pad = vae(padded, torch.tensor([64]), sample=False)
    torch.testing.assert_close(with_pad["mel_hat"][..., :64], alone["mel_hat"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(with_pad["mu"][..., :16], alone["mu"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(with_pad["recon"], alone["recon"], atol=1e-6, rtol=1e-5)
    assert torch.all(with_pad["mel_hat"][..., 64:] == 0)


def test_latent_normalisation_roundtrip():
    vae = small_vae()
    vae.set_latent_stats(torch.full((8,), 0.5), torch.full((8,), 2.0))
    mel = torch.randn(1, 96, 32)
    mu_raw, _ = vae.encode(mel)
    mu_norm, _ = vae.encode(mel, normalize=True)
    torch.testing.assert_close(mu_norm, (mu_raw - 0.5) / 2.0)
    torch.testing.assert_close(vae.decode(mu_norm, normalized=True), vae.decode(mu_raw))


def test_collate_pads_to_multiple():
    items = [{"mel": torch.ones(96, 61), "length": 61, "path": "a"}, {"mel": torch.ones(96, 40), "length": 40, "path": "b"}]
    batch = collate_mels(items, multiple_of=4)
    assert batch["mel"].shape == (2, 96, 64)
    assert batch["lengths"].tolist() == [61, 40]
    assert batch["mel"][1, :, 40:].abs().sum() == 0
