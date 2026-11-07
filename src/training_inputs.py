"""Restore the bundled original training tables and verify their exact bytes."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_training_inputs(data_directory, destination=None):
    data_directory = Path(data_directory)
    metadata = json.loads((data_directory / 'training_inputs_metadata.json').read_text(encoding='utf-8'))
    destination = Path(destination) if destination is not None else data_directory / 'training'
    for name in metadata['files']:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe path in the training input inventory')
        path = destination / name
        if path.exists() and (not path.is_file() or sha256(path) != metadata['files'][name]['sha256']):
            raise ValueError(f'Training input has changed: {name}. Use a fresh input folder.')
    missing = [name for name in metadata['files'] if not (destination / name).is_file()]
    if missing:
        archives = metadata.get('archives') or [{'file': metadata['archive'], 'sha256': metadata['archive_sha256']}]
        with ExitStack() as stack:
            members = {}
            for item in archives:
                archive_path = data_directory / item['file']
                if sha256(archive_path) != item['sha256']:
                    raise ValueError('The training input archive has changed. Restore the supplied copy.')
                archive = stack.enter_context(zipfile.ZipFile(archive_path))
                for name in archive.namelist():
                    if name in members:
                        raise ValueError('Repeated file in the training archives')
                    members[name] = archive
            if set(members) != set(metadata['files']):
                raise ValueError('The training archive differs from its file inventory')
            for name in missing:
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with members[name].open(name) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)
                if sha256(path) != metadata['files'][name]['sha256']:
                    raise ValueError(f'Extracted training input failed its checksum: {name}')
    original_annotation = destination / 'final_gnn_dataset.csv'
    annotation = destination / 'gene_annotations.csv'
    if annotation.exists() and sha256(annotation) != sha256(original_annotation):
        raise ValueError('gene_annotations.csv differs from the bundled annotation table')
    if original_annotation.exists() and not annotation.exists():
        shutil.copyfile(original_annotation, annotation)
    return destination.resolve(), {'status': 'passed', 'required_files_checked': len(metadata['files']),
        'all_required_hashes_matched': True, 'individual_prenatal_data_included': False,
        'historical_mapping_hash_recorded': False,
        'mapping_provenance': metadata['mapping_provenance']}
