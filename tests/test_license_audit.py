import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_licenses import PACKAGES, MODEL_WEIGHT_LICENSES, get_installed_license

def test_license_audit_packages():
    assert "torch" in PACKAGES
    assert "demucs" in PACKAGES
    assert "basic-pitch" in PACKAGES
    assert "all-in-one-infer (candidate)" in MODEL_WEIGHT_LICENSES

def test_license_audit_installed_lookup():
    lic = get_installed_license("torch")
    assert "BSD" in lic or lic != "NOT INSTALLED"
