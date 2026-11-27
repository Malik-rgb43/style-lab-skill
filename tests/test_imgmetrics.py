import shutil

import numpy as np
import pytest

import imgmetrics as im
from fixtures.make_media import make_image

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


def test_stats_known_images(tmp_path):
    grey = np.full((120, 160, 3), 0.5, np.float32)
    s = im.look_stats(im.decode_image(make_image(tmp_path / "g.png", grey)))
    assert s["contrast"] < 0.01 and s["grain"] < 0.002
    noisy = np.clip(grey + np.random.default_rng(1).normal(0, 0.05, grey.shape), 0, 1)
    assert im.look_stats(im.decode_image(make_image(tmp_path / "n.png", noisy)))["grain"] > 0.02
    yy, xx = np.mgrid[0:120, 0:160]
    r = np.hypot((yy - 60) / 60, (xx - 80) / 80)
    vig = grey * (1 - 0.6 * np.clip(r / 1.4, 0, 1) ** 2)[..., None]
    assert im.look_stats(im.decode_image(make_image(tmp_path / "v.png", vig)))["vignette"] < 0.8


def test_palette_distance(tmp_path):
    a = np.zeros((80, 80, 3), np.uint8)
    a[:, :40] = (0, 135, 84)
    a[:, 40:] = (241, 128, 68)
    s = im.look_stats(im.decode_image(make_image(tmp_path / "two.png", a)))
    assert im.palette_distance(s["palette"], ["#008754", "#f18044"]) < 3
    assert im.palette_distance(s["palette"], ["#0000ff"]) > 20


def test_ssim_identity_and_change(tmp_path):
    rng = np.random.default_rng(3)
    a = np.zeros((96, 96, 3), np.float32)
    a[10:40, 10:40] = 1.0
    a[60:90, 50:80] = 0.6
    a += rng.normal(0, 0.01, a.shape).astype(np.float32)
    b = np.zeros_like(a)
    b[50:80, 55:90] = 1.0
    b[5:30, 5:60] = 0.3
    assert im.ssim(a, a) > 0.99
    assert im.ssim(a, b) < 0.6
