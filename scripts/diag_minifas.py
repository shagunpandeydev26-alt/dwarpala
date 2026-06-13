"""
Diagnostic: dump the full 3-element softmax vectors for MiniFASNetV2,
MiniFASNetV1SE, and their sum, for all 4 test images. No conclusions —
just raw numbers, using the CURRENT (unchanged) analyzer code.
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

from dwarpala.prana.minifas_analyzer import (
    _build_minifas_model,
    _load_minifas_weights,
    MiniFASAnalyzer,
)
from dwarpala.utils.model_manager import ModelManager

np.set_printoptions(precision=6, suppress=True)

TEST_IMAGES = Path(__file__).parent.parent / "tests" / "testimg"
NAMES = ["selfie1", "selfie2", "printed", "screencapture"]

# Load weights
mgr = ModelManager()
v2_path = mgr.get_model_path("minifas_v2_2_7")
v1se_path = mgr.get_model_path("minifas_v1se_4_0")

m_v2 = _load_minifas_weights(_build_minifas_model("V2"), v2_path).eval()
m_v1se = _load_minifas_weights(_build_minifas_model("V1SE"), v1se_path).eval()

# Detect faces
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def run_model(model, img, bbox, scale):
    tensor = MiniFASAnalyzer.preprocess(img, bbox, scale=scale)
    t = torch.from_numpy(tensor)
    with torch.no_grad():
        out = model(t).cpu().numpy()[0]
    return out, softmax(out)


def strict_check():
    """Confirm the faithful port loads the weights with strict=True."""
    import torch
    for arch, p in [("V2", v2_path), ("V1SE", v1se_path)]:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        first = next(iter(sd))
        if first.startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        m = _build_minifas_model(arch)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"  {arch}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"    missing[:6]={list(missing)[:6]}")
        if unexpected:
            print(f"    unexpected[:6]={list(unexpected)[:6]}")


print(f"\n{'='*78}")
print("Strict state-dict key check (faithful port)")
print(f"{'='*78}")
strict_check()

print(f"\n{'='*78}")
print("MiniFASNet 12-vector diagnostic (faithful reference port)")
print(f"{'='*78}")

for name in NAMES:
    path = TEST_IMAGES / f"{name}.jpeg"
    img = cv2.imread(str(path))  # BGR
    faces = app.get(img)
    if not faces:
        print(f"\n{name}: NO FACE DETECTED")
        continue
    f = faces[0]
    x1, y1, x2, y2 = f.bbox.astype(int)
    bbox = (x1, y1, x2 - x1, y2 - y1)

    v2_logits, v2_sm = run_model(m_v2, img, bbox, 2.7)
    v1se_logits, v1se_sm = run_model(m_v1se, img, bbox, 4.0)
    summed = v2_sm + v1se_sm

    print(f"\n--- {name}  bbox(xywh)={bbox} ---")
    print(f"  V2   logits = {v2_logits}")
    print(f"  V2   softmax= {v2_sm}   argmax={v2_sm.argmax()}")
    print(f"  V1SE logits = {v1se_logits}")
    print(f"  V1SE softmax= {v1se_sm}   argmax={v1se_sm.argmax()}")
    print(f"  SUM  softmax= {summed}   argmax={summed.argmax()}")
