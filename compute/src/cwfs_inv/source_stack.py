"""Load the material configuration shipped with this package."""
import json
from pathlib import Path
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/material.json"

def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
