import json
import subprocess

import pytest

import doctor
import store


CALLS = []


class FakePopen:
    out, rc, timeout = "tool 1.0\nline2", 0, False

    def __init__(self, cmd, **kw):
        CALLS.append((list(cmd), kw))
        self.pid, self.returncode = 1, type(self).rc

    def communicate(self, timeout=None):
        if type(self).timeout and timeout == 45:
            type(self).timeout = False
            raise subprocess.TimeoutExpired("npx", timeout)
        return type(self).out, ""

    def kill(self):
        pass


@pytest.fixture
def everything(monkeypatch):
    CALLS.clear()
    FakePopen.out, FakePopen.rc, FakePopen.timeout = "tool 1.0\nline2", 0, False
    monkeypatch.setattr(doctor.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(doctor, "_kill_tree", lambda p: None)
    monkeypatch.setattr(doctor.shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda n: object())
    monkeypatch.setenv("RUNPOD_API_KEY", "x")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "y")


def fake_models(tmp_path, monkeypatch, which=("model_wav2vec2_he", "model_whisper_he",
                                              "model_mms_fa", "model_whisper_multi"), env="HF_HOME"):
    hf, th = tmp_path / "hf", tmp_path / "torch"
    for k in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    if env == "HF_HOME":
        monkeypatch.setenv("HF_HOME", str(hf))
        hub = hf / "hub"
    else:
        monkeypatch.setenv("HF_HUB_CACHE", str(hf))
        hub = hf
    monkeypatch.setenv("TORCH_HOME", str(th))
    for n in which:
        kind, mid = doctor.MODELS[n]
        if kind == "hf":
            snap = hub / ("models--" + mid.replace("/", "--")) / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
        else:
            d = th / "hub" / "checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            (d / __import__("os").path.basename(__import__("align").MMS_URL)).write_text("x")


def caps():
    return store.read_json(store.path("capabilities.json"), None)


def test_all_present(home, everything, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    assert doctor.main([]) == 0
    c = caps()
    assert c["checked"]
    assert set(c["stages"]) == set(doctor.STAGES)
    assert all(s["missing"] == [] for s in c["stages"].values())


def test_missing_ffmpeg(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "ffmpeg" else f"/bin/{n}")
    doctor.main([])
    c = caps()
    assert c["stages"]["train"]["missing"] == ["ffmpeg"]
    assert c["local"]["ffmpeg"]["hint"]


def test_hyperframes_timeout(home, everything, monkeypatch):
    FakePopen.timeout = True
    doctor.main([])
    h = caps()["local"]["hyperframes"]
    assert h["ok"] is False and h["note"] == "check timed out"


def test_hyperframes_noninteractive(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "hyperframes" else f"/bin/{n}")
    doctor.main([])
    cmd, kw = next(c for c in CALLS if "hyperframes" in c[0])
    assert "--no" in cmd and "--yes" not in cmd and kw["stdin"] == subprocess.DEVNULL
    assert caps()["local"]["hyperframes"]["ok"] is True


def test_hyperframes_npx_not_cached_never_downloads(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "hyperframes" else f"/bin/{n}")
    FakePopen.rc = 1
    h = doctor.check_hyperframes()
    cmd = next(c for c, _ in CALLS if "hyperframes" in c)
    assert cmd[1] == "--no" and "--yes" not in cmd
    assert h["ok"] is False and h["note"].startswith("not installed")
    assert "ask the user first" in h["note"]


def test_hyperframes_failed_note(home, everything, monkeypatch):
    FakePopen.rc = 3  # a global hyperframes binary that fails its --version
    h = doctor.check_hyperframes()
    assert h["note"] == "check failed (exit 3)"


def test_blender_found_in_program_files(home, everything, monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "blender" else f"/bin/{n}")
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    for v in ("4.2", "5.2"):
        d = tmp_path / "Blender Foundation" / f"Blender {v}"
        d.mkdir(parents=True)
        (d / "blender.exe").write_text("x")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    doctor.main([])
    b = caps()["local"]["blender"]
    assert b["ok"] is True and "Blender 5.2" in b["path"]
    assert caps()["local"]["ffmpeg"]["path"] == "/bin/ffmpeg"


def test_torch_cuda_not_probed_by_default(home, everything):
    doctor.main([])
    assert caps()["local"]["torch"]["cuda"] == "not probed"


def test_env_values_never_written(home, everything, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret123")
    doctor.main([])
    assert "secret123" not in store.path("capabilities.json").read_text(encoding="utf-8")
    assert caps()["stages"]["cloud_asr"]["paid"] == ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]


def test_session_set_preserved(home, everything):
    assert doctor.main(["session", "--set", "higgsfield=ok", "--note", "mcp"]) == 0
    doctor.main([])
    s = caps()["session"]["higgsfield"]
    assert s["ok"] is True and s["note"] == "mcp"


def test_session_clear_and_bad_value(home, everything):
    doctor.main(["session", "--set", "pexels=ok"])
    assert doctor.main(["session", "--set", "pexels=maybe"]) == 1
    doctor.main(["session", "--clear"])
    assert caps()["session"] == {}


def test_json_output(home, everything, capsys):
    doctor.main(["--json"])
    assert "stages" in json.loads(capsys.readouterr().out)


def test_models_missing_report_hint(home, everything, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch, which=("model_whisper_he",))
    doctor.main([])
    c = caps()
    assert c["local"]["model_whisper_he"]["ok"] is True
    assert c["local"]["model_wav2vec2_he"]["ok"] is False
    assert "--allow-download" in c["local"]["model_wav2vec2_he"]["hint"]
    assert c["stages"]["cut"]["missing"] == ["model_wav2vec2_he"]
    assert c["stages"]["cut_other_languages"]["missing"] == ["model_mms_fa", "model_whisper_multi"]


def test_cut_stage_uses_stack(home, everything, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    doctor.main([])
    uses = caps()["stages"]["cut"]["uses"]
    for n in ("ffmpeg", "torch", "torchaudio", "transformers", "faster_whisper"):
        assert n in uses


def test_model_check_never_loads(home, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch, which=("model_mms_fa",))
    assert doctor.check_model("model_mms_fa")["ok"] is True
    assert doctor.check_model("model_whisper_multi")["ok"] is False


def test_models_found_via_hf_hub_cache(home, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch, which=("model_whisper_he",), env="HF_HUB_CACHE")
    assert doctor.check_model("model_whisper_he")["ok"] is True
    assert doctor.check_model("model_wav2vec2_he")["ok"] is False


def test_empty_model_folder_is_missing(home, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch, which=())
    (tmp_path / "hf" / "hub" / "models--ivrit-ai--whisper-large-v3-turbo-ct2").mkdir(parents=True)
    assert doctor.check_model("model_whisper_he")["ok"] is False


def _style_with(home, tmp_path, *reqs):
    import styles
    styles.main(["add", "--name", "Demo", "--pack", str(tmp_path)])
    for name, level in reqs:
        styles.main(["require", "demo", "--name", name, "--why", "needed", "--level", level,
                     "--fallback", "do it by hand"])


def test_style_missing_required_exit_4(home, everything, tmp_path, monkeypatch, capsys):
    fake_models(tmp_path, monkeypatch)
    _style_with(home, tmp_path, ("higgsfield", "required"))
    assert doctor.main(["--style", "demo"]) == 4
    out = capsys.readouterr().out
    assert "higgsfield" in out and "do it by hand" in out
    doctor.main(["session", "--set", "higgsfield=ok"])
    assert doctor.main(["--style", "demo"]) == 0
    assert "✓ higgsfield" in capsys.readouterr().out


def test_style_recommended_missing_exit_0_with_note(home, everything, tmp_path, monkeypatch, capsys):
    fake_models(tmp_path, monkeypatch)
    _style_with(home, tmp_path, ("pexels", "recommended"), ("ffmpeg", "required"))
    assert doctor.main(["--style", "demo"]) == 0
    out = capsys.readouterr().out
    assert "חסר pexels" in out and "only recommended" in out and "✓ ffmpeg" in out


def test_style_unknown(home, everything, tmp_path, monkeypatch):
    fake_models(tmp_path, monkeypatch)
    assert doctor.main(["--style", "nope"]) == 1


def test_check_model_missing_numpy_reported_not_crash(home, monkeypatch):
    import builtins
    import sys as _sys
    monkeypatch.delitem(_sys.modules, "align", raising=False)
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "align":
            raise ImportError("No module named 'numpy'", name="numpy")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    r = doctor.check_model("model_whisper_he")
    assert r["ok"] is False and "numpy" in r["note"] and "pip install numpy" in r["hint"]


def test_train_stage_reports_optional(home, everything, monkeypatch, capsys):
    """cv2, librosa and fontTools missing: train shows them as recommended/optional with hints and
    does not fail (only ffmpeg, ffprobe and numpy are required)."""
    gone = {"cv2", "librosa", "fontTools"}
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda n: None if n in gone else object())
    assert doctor.main([]) == 0
    c = caps()
    t = c["stages"]["train"]
    assert t["missing"] == []
    assert t["recommended_missing"] == ["cv2"]
    assert {"librosa", "fontTools"} <= set(t["optional_missing"])
    for n in ("cv2", "librosa", "fontTools"):
        assert c["local"][n]["hint"]
    out = capsys.readouterr().out
    assert "cv2" in out and "librosa" in out
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "ffmpeg" else f"/bin/{n}")
    doctor.main([])
    assert caps()["stages"]["train"]["missing"] == ["ffmpeg"]


def test_ocr_and_yunet_checks(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.importlib.util, "find_spec",
                        lambda n: None if n == "rapidocr_onnxruntime" else object())
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "tesseract" else f"/bin/{n}")
    doctor.main([])
    c = caps()
    assert c["local"]["ocr"]["ok"] is False and c["local"]["ocr"]["hint"]
    assert c["local"]["model_yunet"]["ok"] is False and "fetch_model.py" in c["local"]["model_yunet"]["hint"]
    p = store.path("models", doctor.YUNET_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 10)
    doctor.main([])
    assert caps()["local"]["model_yunet"]["ok"] is True


def test_motion_and_ideas_stages(home, everything, monkeypatch):
    def which(n):
        return None if n in ("hyperframes",) else f"/bin/{n}"
    monkeypatch.setattr(doctor.shutil, "which", which)
    FakePopen.rc = 1  # npx --no hyperframes fails: not cached
    doctor.main([])
    st = caps()["stages"]
    assert "hyperframes" in st["motion"]["missing"]
    assert st["motion"]["uses"] == ["node", "hyperframes", "ffmpeg", "ffprobe"]
    assert st["ideas"]["uses"] == ["ffmpeg", "ffprobe"] and st["ideas"]["missing"] == []


def test_cloud_asr_preapproved_setting(home):
    import store
    assert store.settings()["cloud_asr_preapproved"] is False


def test_train_profile_instaloader_optional(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda n: None if n == "instaloader" else object())
    doctor.main([])
    st = caps()["stages"]["train_profile"]
    assert st["missing"] == [] and st["optional_missing"] == ["instaloader"]
    assert "instaloader" in caps()["local"]["instaloader"]["hint"]
