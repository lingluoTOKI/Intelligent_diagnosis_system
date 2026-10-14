"""Refresh course evidence without modifying images, weights, notebooks or history."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/evidence"
CLASSES = ("A", "C", "D", "G", "H", "M", "N", "O")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(path: Path) -> str:
    with path.open("rb") as file:
        result = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
        return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-structures", action="store_true", help="Inspect trusted local checkpoints with the local PyTorch/YOLO environment")
    args = parser.parse_args()
    result = {"audited_at": datetime.now().astimezone().isoformat(), "counts": {}, "missing_classes": [],
              "scope": "Content hashes detect byte-identical images only; patient IDs and label conversion are unverified."}
    hashed = {}
    for split in ("train", "val"):
        counts, hashes = {}, {}
        for label in CLASSES:
            folder = ROOT / "data" / split / label
            if not folder.is_dir():
                result["missing_classes"].append(f"{split}/{label}")
            images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            counts[label] = len(images)
            for image in images:
                hashes.setdefault(digest(image), []).append(image.relative_to(ROOT).as_posix())
        result["counts"][split] = counts
        hashed[split] = hashes
    overlaps = sorted(set(hashed["train"]) & set(hashed["val"]))
    result["identical_file_hashes_across_splits"] = len(overlaps)
    result["duplicate_evidence"] = [{"sha256": h, "train": hashed["train"][h], "val": hashed["val"][h]} for h in overlaps]
    summary = ROOT / "runs/comparison_summary.csv"
    result["experiments"] = []
    if summary.is_file():
        with summary.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                source = ROOT / Path(row["source_csv"].replace("\\", "/"))
                record = dict(row, source_exists=source.is_file())
                if source.is_file():
                    with source.open(encoding="utf-8-sig", newline="") as logs:
                        records = list(csv.DictReader(logs))
                    record["logged_epochs"] = len(records)
                    record["source_sha256"] = digest(source)
                result["experiments"].append(record)
    if args.model_structures:
        sys.path.insert(0, str(ROOT / "ultralytics-main"))
        from ultralytics import YOLO
        result["structure_audit"] = []
        paths = [("deployed", ROOT / "models/custom/AKConv_best_moudle/best.pt")]
        paths += [(row["model"], ROOT / Path(row["source_csv"].replace("\\", "/")).parent / "weights/best.pt")
                  for row in result["experiments"]]
        for label, path in paths:
            model = YOLO(str(path)).model
            structure = [{"name": name, "type": type(layer).__name__,
                          "parameters": [(k, list(p.shape)) for k, p in layer.named_parameters(recurse=False)]}
                         for name, layer in model.named_modules()]
            tensors = hashlib.sha256()
            for name, value in model.state_dict().items():
                tensors.update(name.encode("utf-8"))
                tensors.update(value.detach().cpu().numpy().tobytes())
            result["structure_audit"].append({"model": label, "weight": path.relative_to(ROOT).as_posix(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "akconv_layers": sum(type(m).__name__ == "AKConv" for m in model.modules()),
                "structure_sha256": hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest(),
                "state_tensors_sha256": tensors.hexdigest(), "preprocessing": str(getattr(model, "transforms", None)),
                "inspection_loader": "project local Ultralytics; training framework versions remain as recorded in manifests"})
    result["model_files"] = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
                             for p in (ROOT / "models").rglob("*") if p.is_file() and p.suffix in {".pt", ".onnx"}}
    result["server_health"] = []
    for address in ("http://101.132.104.41:8000/api/health", "http://game.lingluotoki.dpdns.org:8000/api/health"):
        observation = {"url": address, "tested_at": datetime.now().astimezone().isoformat()}
        try:
            import requests
            with requests.Session() as session:
                # Public health checks should not be confused with failures of a local desktop proxy.
                session.trust_env = False
                response = session.get(address, timeout=12)
            observation["transport"] = "direct; environment proxy disabled for this request only"
            observation.update(http_status=response.status_code, response=response.json())
        except Exception as error:
            observation["error"] = str(error)
        result["server_health"].append(observation)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "course_evidence_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "totals": {k: sum(v.values()) for k, v in result["counts"].items()},
                      "duplicate_hashes": len(overlaps), "experiments": len(result["experiments"]),
                      "server_http": [r.get("http_status", r.get("error")) for r in result["server_health"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
