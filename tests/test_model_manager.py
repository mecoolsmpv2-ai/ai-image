from __future__ import annotations

from pathlib import Path

from comfyui_app import model_manager
from comfyui_app.model_resolver import ModelResolverError


def _write_file(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_list_installed_models_reports_sizes_and_total(tmp_path: Path, monkeypatch) -> None:
    comfyui_dir = tmp_path / "ComfyUI"
    models_dir = comfyui_dir / "models"
    _write_file(models_dir / "diffusion_models" / "int8.safetensors", b"12345")
    _write_file(models_dir / "vae" / "vae.pth", b"123")
    _write_file(comfyui_dir / "custom_nodes" / "comfyui_controlnet_aux" / "ckpts" / "depth_anything_v2_vitl.pth", b"1234")

    monkeypatch.setattr(model_manager, "COMFYUI_DIR", comfyui_dir)
    monkeypatch.setattr(model_manager, "MODELS_DIR", models_dir)

    payload = model_manager.list_installed_models()
    labels = {entry["label"] for entry in payload["entries"]}

    assert payload["count"] == 3
    assert payload["total_bytes"] == 12
    assert payload["total"] == "12 B"
    assert any(label.startswith("diffusion_models/int8.safetensors") for label in labels)
    assert any(label.startswith("vae/vae.pth") for label in labels)
    assert any("comfyui_controlnet_aux" in entry["category"] for entry in payload["entries"])


def test_delete_models_removes_files_and_prunes_manifest(tmp_path: Path, monkeypatch) -> None:
    comfyui_dir = tmp_path / "ComfyUI"
    models_dir = comfyui_dir / "models"
    target = _write_file(models_dir / "diffusion_models" / "int8.safetensors", b"12345")
    written_manifests: list[dict[str, object]] = []
    manifest = {
        "engine": "int8",
        "timestamp": "2024-01-01T00:00:00",
        "models": {
            "diffusion": {
                "dest_dir": str(models_dir / "diffusion_models"),
                "local_filename": "int8.safetensors",
                "local_path": str(target),
            }
        },
    }

    monkeypatch.setattr(model_manager, "COMFYUI_DIR", comfyui_dir)
    monkeypatch.setattr(model_manager, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_manager, "load_resolved_manifest", lambda: manifest)
    monkeypatch.setattr(model_manager, "_write_manifest", lambda data: written_manifests.append(data))

    payload = model_manager.delete_models([target])

    assert not target.exists()
    assert payload["freed_bytes"] == 5
    assert payload["total_bytes"] == 0
    assert written_manifests
    assert written_manifests[0]["models"] == {}


def test_delete_models_rejects_outside_tree(tmp_path: Path, monkeypatch) -> None:
    comfyui_dir = tmp_path / "ComfyUI"
    models_dir = comfyui_dir / "models"
    _write_file(models_dir / "diffusion_models" / "int8.safetensors", b"12345")
    outside = _write_file(tmp_path / "outside.safetensors", b"999")

    monkeypatch.setattr(model_manager, "COMFYUI_DIR", comfyui_dir)
    monkeypatch.setattr(model_manager, "MODELS_DIR", models_dir)

    try:
        model_manager.delete_models([outside])
        raise AssertionError("Expected ModelResolverError")
    except ModelResolverError as exc:
        assert "outside the ComfyUI model tree" in exc.message
    assert outside.exists()
