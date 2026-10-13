"""Check the configured classification runtime without GUI, network API calls, or history writes."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import os
import sys

os.environ["MPLBACKEND"] = "Agg"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.runtime_config import load_app_config, project_path
from configs.network_config import get_ipv4_interfaces, get_local_ip
from src.core.classification import EyeDiseaseDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional local image; classification is not saved to history")
    parser.add_argument("--device", help="cpu, auto, or CUDA index such as 0")
    args = parser.parse_args()
    config = load_app_config()
    if args.device:
        config = replace(config, device=args.device)
    detector = EyeDiseaseDetector(config)
    if not detector.load_model():
        print(json.dumps({"success": False, "error": detector.last_error}, ensure_ascii=False))
        return 1
    output = {"success": True, "python": sys.executable, "model": detector.model_info,
              "lan_ip": get_local_ip(), "lan_interfaces": get_ipv4_interfaces()}
    if args.image:
        import cv2
        import numpy as np
        try:
            image = cv2.imdecode(np.fromfile(project_path(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            image = None
        result = detector.classify(image)
        output["classification"] = result.to_dict()
        output["success"] = result.success
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
