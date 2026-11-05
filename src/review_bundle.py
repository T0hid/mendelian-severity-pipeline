"""Export frozen figures, source tables and a local reviewer report.

This module never fits a model or changes an analysis result. Original figure
exports and CSV bytes are retained; Excel workbooks provide a convenient view
of the same source tables.
"""
from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import sys
import zipfile
from pathlib import Path, PurePosixPath
from xml.sax.saxutils import escape


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_assets(data_directory, destination):
    """Validate each archive's inventory and hashes before extracting it."""
    manifest = json.loads((data_directory / 'publication_assets.json').read_text(encoding='utf-8'))
    if destination.is_symlink():
        raise ValueError('Publication output must not be a symbolic link.')
    for record in manifest['archives']:
        archive_path = data_directory / record['file']
        if archive_path.name != record['file'] or sha256(archive_path) != record['sha256']:
            raise ValueError('Publication archive checksum mismatch.')
        expected = {row['path']: row for row in manifest['files'] if row['archive'] == record['file']}
        _extract_checked_archive(archive_path, expected, destination)
    return manifest


def _extract_checked_archive(archive_path, expected, destination):
    with zipfile.ZipFile(archive_path) as archive:
        names = [item.filename for item in archive.infolist()]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Publication archive inventory mismatch.')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe publication archive path.')
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != expected[name]['sha256']:
                raise ValueError('Publication member checksum mismatch: ' + name)
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = destination.joinpath(*PurePosixPath(name).parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Publication destination leaves its output directory.')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def _column_name(number):
    result = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_source_workbook(path, tables):
    """Write values-only OOXML workbooks without additional dependencies.

    CSV files remain authoritative. Text, including formula-like strings, is
    encoded as inline text. Finite numeric fields are numeric Excel cells.
    """
    namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel_namespace = 'http://schemas.openxmlformats.org/package/2006/relationships'
    content_types = [('rels', 'application/vnd.openxmlformats-package.relationships+xml'),
                     ('xml', 'application/xml')]
    workbook_sheets, workbook_relationships, overrides = [], [], []
    seen = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for index, (title, rows) in enumerate(tables, 1):
            title = ''.join('_' if c in '[]:*?/\\' else c for c in title)[:31] or 'Data'
            if title.lower() in seen:
                title = title[:26] + '_' + str(index)
            seen.add(title.lower())
            workbook_sheets.append(f'<sheet name="{escape(title, {chr(34): "&quot;"})}" sheetId="{index}" r:id="rId{index}"/>')
            workbook_relationships.append(f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>')
            overrides.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            xml_rows = []
            for row_number, row in enumerate(rows, 1):
                cells = []
                for column, value in enumerate(row, 1):
                    text = '' if value is None else str(value)
                    address = _column_name(column) + str(row_number)
                    numeric = False
                    if row_number > 1 and text.strip() and not (len(text) > 1 and text.startswith('0') and text[1].isdigit()):
                        try:
                            numeric = math.isfinite(float(text))
                        except ValueError:
                            pass
                    if numeric:
                        cells.append(f'<c r="{address}"><v>{escape(text)}</v></c>')
                    else:
                        cells.append(f'<c r="{address}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>')
                xml_rows.append(f'<row r="{row_number}">' + ''.join(cells) + '</row>')
            width = max((len(row) for row in rows), default=1)
            archive.writestr(f'xl/worksheets/sheet{index}.xml',
                f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{namespace}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="{width}" width="22" customWidth="1"/></cols><sheetData>' + ''.join(xml_rows) + '</sheetData></worksheet>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{namespace}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + ''.join(workbook_sheets) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{rel_namespace}">' + ''.join(workbook_relationships) + '</Relationships>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{rel_namespace}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + ''.join(f'<Default Extension="{ext}" ContentType="{mime}"/>' for ext, mime in content_types) + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + ''.join(overrides) + '</Types>')


def export_source_workbooks(publication, manifest):
    outputs = []
    for figure in manifest['figures']:
        tables = [('Source index', [['Figure', 'Source file', 'Scope']] +
                  [[figure['label'], source, figure['source_scope']] for source in figure['source_tables']])]
        for source in figure['source_tables']:
            with (publication / source).open(encoding='utf-8-sig', newline='') as handle:
                tables.append((Path(source).stem, list(csv.reader(handle))))
        workbook = publication / 'source_data' / (figure['id'] + '.xlsx')
        write_source_workbook(workbook, tables)
        outputs.append(workbook.relative_to(publication).as_posix())
    return outputs


def execution_environment():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'packages': versions, 'model_training_performed': False}


def write_gallery(output, manifest, replay=None):
    cards = []
    for figure in manifest['figures']:
        prefix = 'publication/'
        png = next((p for p in figure['assets'] if p.endswith('.png')), None)
        links = ' '.join(f'<a href="{prefix + html.escape(p, quote=True)}">{Path(p).suffix[1:].upper()}</a>' for p in figure['assets'])
        links += f' <a href="publication/source_data/{figure["id"]}.xlsx">Source data XLSX</a>'
        image = f'<img loading="lazy" src="{prefix + html.escape(png, quote=True)}" alt="{html.escape(figure["label"], quote=True)}">' if png else ''
        cards.append(f'<section><h2>{html.escape(figure["label"])}</h2><p>{html.escape(figure["source_scope"])}</p><p>{links}</p>{image}</section>')
    replay_links = ''
    if replay:
        replay_links = '<h2>Plots rendered during this run</h2><ul>' + ''.join(
            f'<li><a href="{html.escape(p, quote=True)}">{html.escape(Path(p).name)}</a></li>' for p in replay['files']
            if Path(p).suffix in {'.png', '.pdf'}) + '</ul>'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mendelian severity results</title><style>body{font:16px/1.55 Arial,sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#17212b;background:#fff}h1,h2{line-height:1.2}a{color:#075e95}section{margin:48px 0;border-top:1px solid #ddd;padding-top:20px}img{max-width:100%;height:auto;border:1px solid #eee}code{background:#f2f4f5;padding:2px 5px}</style><h1>Mendelian severity results</h1><p>The images below are the original saved exports. Source tables are available as CSV and one Excel workbook per figure. The replay links identify plots rendered from saved results during this run. No model was retrained.</p><p><a href="predictions.csv">Example predictions</a> · <a href="verification.json">Verification report</a> · <a href="review_manifest.json">File provenance</a> · <a href="publication/figure_map.csv">Figure and code map</a></p>'''
    if not (output / 'predictions.csv').is_file():
        document = document.replace('<a href="predictions.csv">Example predictions</a> · ', '')
    if not (output / 'verification.json').is_file():
        document = document.replace('<a href="verification.json">Verification report</a> · ', '')
    (output / 'index.html').write_text(document + replay_links + ''.join(cards) + '</html>', encoding='utf-8')


def build_review_bundle(data_directory, output, replot=True):
    data_directory, output = Path(data_directory), Path(output)
    if output.resolve().is_relative_to(data_directory.resolve()) or data_directory.resolve().is_relative_to(output.resolve()):
        raise ValueError('Reviewer output must be separate from the bundled data directory.')
    output.mkdir(parents=True, exist_ok=True)
    publication = output / 'publication'
    if publication.exists() and any(publication.iterdir()):
        raise ValueError('Reviewer publication output already exists; choose a new --output-dir.')
    manifest = extract_assets(data_directory, publication)
    workbooks = export_source_workbooks(publication, manifest)
    replay = None
    if replot:
        from replot_saved import render_saved_figures
        replay = render_saved_figures(data_directory / 'reference', publication, output / 'replotted')
        replay['files'] = [(Path('replotted') / p).as_posix() for p in replay['files']]
    report = {'status': 'passed', 'scope': 'Original figure export, source-data export and saved-data plotting.',
              'publication_archives': manifest['archives'], 'original_files_exported': len(manifest['files']),
              'source_workbooks': workbooks, 'replay': replay, 'environment': execution_environment(),
              'original_numeric_source_modified': False}
    report['output_files'] = [{'path': p.relative_to(output).as_posix(), 'sha256': sha256(p)}
                              for p in sorted(output.rglob('*')) if p.is_file() and p.name not in {'index.html', 'review_manifest.json'}]
    (output / 'review_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    write_gallery(output, manifest, replay)
    return report
