"""One-time, hash-checked application of the reviewed UI delta."""
import base64
import bz2
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '82c0c822c52f23cb5f192882989a42dc838979ebca1677b917ab831c79a4a7ed'
ALLOWED = {'mesh_gui.py', 'mesh_ui_layout.py', 'mesh_project_io.py',
    'test_ui_workflows.py', 'audit_reproduce.py', 'capture_ui.py',
    'mesh_grid_studio_version.txt', 'UI_REVIEW.md', 'README.md', 'SUPPLEMENTARY.md',
    '.github/workflows/scientific-audit.yml', '.github/workflows/reproduce-research.yml',
    '.github/workflows/windows-ui-preview.yml'}
parts = []
for index in range(6):
    part = (ROOT / f'.github/ui_payload/part-{index:02d}.b64').read_text().strip()
    print(index, len(part), hashlib.sha256(part.encode()).hexdigest(), flush=True)
    parts.append(part)
raw = bz2.decompress(base64.b64decode(''.join(parts), validate=True))
assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Payload SHA-256 mismatch'
payload = json.loads(raw)
assert set(payload) == ALLOWED, 'Unexpected source paths'
outputs = {}
for filename, item in payload.items():
    path = ROOT / filename
    original = path.read_bytes() if path.exists() else None
    actual = hashlib.sha256(original).hexdigest() if original is not None else None
    assert actual == item['before'], f'Baseline changed: {filename}: {actual}'
    lines = original.decode('utf-8').splitlines(keepends=True) if original is not None else []
    previous_end = 0
    for start, end, replacement in item['edits']:
        assert isinstance(start, int) and isinstance(end, int)
        assert previous_end <= start <= end <= len(lines)
        assert isinstance(replacement, str)
        previous_end = end
    for start, end, replacement in reversed(item['edits']):
        lines[start:end] = replacement.splitlines(keepends=True)
    result = ''.join(lines).encode('utf-8')
    assert hashlib.sha256(result).hexdigest() == item['after'], f'Result hash mismatch: {filename}'
    outputs[path] = result
for path, result in outputs.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result)
    print('Verified and applied:', path.relative_to(ROOT), flush=True)
