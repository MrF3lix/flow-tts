import torch

from test_tts.data.lengths import sequence_mask
from test_tts.models.latent_fm import LatentFlowTTS

N_MELS, C, R = 12, 4, 4


def small_model(**kw):
    torch.manual_seed(0)
    return LatentFlowTTS(
        n_vocab=20, n_mels=N_MELS, latent_dim=C, compression=R,
        encoder=dict(hidden=32, n_layers=2, n_heads=2, prenet_layers=2, dropout=0.0),
        duration_predictor=dict(filter_channels=16, dropout=0.0),
        decoder=dict(d_model=32, depth=2, n_heads=2, dropout=0.0, conv_pos_kernel=5),
        **kw,
    ).eval()


def make_batch():
    g = torch.Generator().manual_seed(0)
    x_lengths = torch.tensor([9, 6])
    mel_lengths = torch.tensor([37, 22])  # not multiples of R on purpose
    z_lengths = -(-mel_lengths // R)
    L = int(z_lengths.max())
    x = torch.randint(3, 20, (2, 9), generator=g) * sequence_mask(x_lengths, 9)
    mel = torch.randn(2, N_MELS, L * R, generator=g) * sequence_mask(mel_lengths, L * R)[:, None]
    z = torch.randn(2, C, L, generator=g) * sequence_mask(z_lengths, L)[:, None]
    return {"x": x, "x_lengths": x_lengths, "mel": mel, "mel_lengths": mel_lengths, "z": z, "z_lengths": z_lengths}


def test_training_step_losses_and_gradients():
    model = small_model().train()
    out = model.training_step(make_batch())
    assert set(out) == {"loss", "fm", "prior", "dur"}
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_encoder_padding_invariance():
    model = small_model()
    x = torch.randint(3, 20, (1, 7))
    h1, mu1, _ = model.encoder(x, torch.tensor([7]))
    h2, mu2, _ = model.encoder(torch.nn.functional.pad(x, (0, 5)), torch.tensor([7]))
    torch.testing.assert_close(h2[:, :7], h1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mu2[..., :7], mu1, atol=1e-5, rtol=1e-5)
    assert torch.all(h2[:, 7:] == 0)


def test_decoder_padding_invariance():
    model = small_model()
    for p in model.decoder.parameters():  # adaLN-zero/zero-init output would make this trivial
        torch.nn.init.normal_(p, 0, 0.1)
    z, cond, t = torch.randn(1, C, 10), torch.randn(1, 32, 10), torch.tensor([0.3])
    v1 = model.decoder(z, t, cond, torch.ones(1, 10, dtype=torch.bool))
    pad = lambda a: torch.nn.functional.pad(a, (0, 6))  # noqa: E731
    v2 = model.decoder(pad(z), t, pad(cond), sequence_mask(torch.tensor([10]), 16))
    torch.testing.assert_close(v2[..., :10], v1, atol=1e-5, rtol=1e-5)
    assert torch.all(v2[..., 10:] == 0)


def test_validation_loss_is_deterministic_with_fixed_noise():
    model, batch = small_model(), make_batch()
    noise, t = torch.randn_like(batch["z"]), torch.rand(2)
    a = model.training_step(batch, noise=noise, t=t, cond_drop=False)
    b = model.training_step(batch, noise=noise, t=t, cond_drop=False)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_synthesise_shapes_and_oracle_durations():
    model, batch = small_model(), make_batch()
    out = model.synthesise(batch["x"], batch["x_lengths"], n_steps=4, guidance_scale=2.0,
                           generator=torch.Generator().manual_seed(0))
    B, _, L = out["z"].shape
    assert B == 2 and L == int(out["z_lengths"].max())
    assert torch.all(out["z_lengths"] == -(-out["mel_lengths"] // R))
    assert torch.all(out["z"][1, :, int(out["z_lengths"][1]):] == 0)

    durations = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 4, 5], [5, 3, 3, 3, 4, 4, 0, 0, 0]])
    out = model.synthesise(batch["x"], batch["x_lengths"], n_steps=2, durations=durations)
    assert out["mel_lengths"].tolist() == [37, 22]
    assert out["z_lengths"].tolist() == [10, 6]
    torch.testing.assert_close(out["path"].sum(-1), durations.float())
