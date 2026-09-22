"""Fail on mismatched scientific tables, saved-grid metrics, source hashes or ZIP contents."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import mesh_methods as m2
import mesh_methods_3d as m3
from audit_reproduce import generate_discussion
from build_supplementary import canonical_bytes

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/generated'


def verify() -> None:
    for stem, writer in (('results', m2.write_latex_table), ('results_3d', m3.write_latex_table_3d)):
        rows = list(csv.DictReader((OUT/f'{stem}.csv').open(encoding='utf-8-sig')))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'table.tex'
            writer(rows, path)
            expected = 'results_table_3d.tex' if stem.endswith('_3d') else 'results_table.tex'
            if path.read_bytes() != (OUT/expected).read_bytes():
                raise AssertionError(f'{expected} does not match its CSV')
    discussion = (OUT/'discussion_3d.tex').read_bytes()
    # Generate into the existing path but restore it even when the check fails.
    try:
        generate_discussion()
        if (OUT/'discussion_3d.tex').read_bytes() != discussion:
            raise AssertionError('Discussion numbers do not match CSV')
    finally:
        (OUT/'discussion_3d.tex').write_bytes(discussion)
    rows = list(csv.DictReader((OUT/'results_3d.csv').open(encoding='utf-8-sig')))
    metadata = json.loads((OUT/'run_metadata_3d.json').read_text())
    if len(rows) != 12 or len(metadata['runs']) != 12:
        raise AssertionError('Expected the 12-row main 3D experiment')
    for row, run in zip(rows, metadata['runs']):
        if (row['domain'], row['method']) != (run['domain'], run['method']):
            raise AssertionError('CSV and metadata identify different runs')
        grid = np.load(OUT/run['grid_file'], allow_pickle=False)['grid']
        result = m2.GridResult(run['method'], grid, run['converged'], run['iterations'],
                               run['residual'], run['runtime_s'], parameters=run['parameters'])
        actual = m3.grid_metrics_3d(result)
        for k in ('volume', 'volume_cv', 'aspect_p95', 'orthogonality_score',
                  'min_scaled_jacobian', 'inverted_cells', 'min_sampled_jacobian', 'runtime_ms'):
            if not np.isclose(float(row[k]), float(actual[k]), rtol=2e-11, atol=2e-12):
                raise AssertionError(f'{run["grid_file"]}: {k} differs from CSV')
        if not run['converged']:
            raise AssertionError(f'Main-series run did not converge: {run["grid_file"]}')
    evidence = json.loads((ROOT/'output/audit/verification.json').read_text())
    if not evidence['tests_passed']:
        raise AssertionError('Recorded unit tests failed')
    for filename, digest in evidence['source_sha256'].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Source changed since tests: {filename}')
    with zipfile.ZipFile(ROOT/'output/article-supplementary-materials.zip') as archive:
        manifest = json.loads(archive.read('MANIFEST_SHA256.json'))
        required = {'mesh_methods_3d.py', 'mesh_geometry_3d.py', 'test_scientific_regressions.py',
                    'article.tex', 'output/generated/results_3d.csv', 'output/pdf/article.pdf'}
        if not required.issubset(manifest):
            raise AssertionError('Supplementary ZIP is missing required 3D files')
        for name, digest in manifest.items():
            if '\\' in name or name.startswith('/') or '..' in Path(name).parts:
                raise AssertionError(f'Nonportable/unsafe archive path: {name}')
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise AssertionError(f'ZIP hash differs: {name}')
            if canonical_bytes(ROOT/name) != archive.read(name):
                raise AssertionError(f'ZIP is stale relative to source: {name}')
    print('PASS: CSV/TeX tables, generated discussion, 12 saved-grid metrics, source hashes, supplementary manifest')


if __name__ == '__main__':
    verify()
