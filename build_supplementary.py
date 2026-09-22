"""Build a self-contained, root-layout research ZIP with a SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
TARGET = ROOT/'output/article-supplementary-materials.zip'


def canonical_bytes(path: Path) -> bytes:
    """Canonical LF bytes make archives independent of Git checkout line endings."""
    content = path.read_bytes()
    if path.suffix.lower() in {'.py', '.tex', '.md', '.txt', '.ps1', '.json', '.csv'}:
        content = content.replace(b'\r\n', b'\n')
    return content


def files_to_package() -> list[Path]:
    files = []
    for pattern in ('*.py', '*.tex', '*.md', 'requirements*.txt', '*.ps1', 'mesh_grid_studio_version.txt'):
        files.extend(ROOT.glob(pattern))
    for directory in ('tests/data', 'docs', 'output/generated', 'output/audit'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    pdf = ROOT/'output/pdf/article.pdf'
    if not pdf.exists():
        raise FileNotFoundError('Build output/pdf/article.pdf before packaging')
    files.append(pdf)
    return sorted(set(files))


def main() -> None:
    files = files_to_package()
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(canonical_bytes(p)).hexdigest() for p in files}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TARGET, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in files:
            archive.writestr(p.relative_to(ROOT).as_posix(), canonical_bytes(p))
        archive.writestr('MANIFEST_SHA256.json', json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(f'Wrote {TARGET.name}: {len(files)} files with repository-root paths')


if __name__ == '__main__':
    main()
