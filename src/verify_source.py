"""Check retained computational definitions against the original-source fingerprints."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from runtime_paths import reference_directory


def canonical_hash(node: ast.AST, renames: dict[str, str]) -> str:
    class Normalise(ast.NodeTransformer):
        def visit_Name(self, item):
            item.id = renames.get(item.id, item.id)
            return item

        def visit_FunctionDef(self, item):
            item.name = renames.get(item.name, item.name)
            self.generic_visit(item)
            self.remove_docstring(item)
            return item

        def visit_ClassDef(self, item):
            self.generic_visit(item)
            self.remove_docstring(item)
            return item

        @staticmethod
        def remove_docstring(item):
            if (item.body and isinstance(item.body[0], ast.Expr)
                    and isinstance(item.body[0].value, ast.Constant)
                    and isinstance(item.body[0].value.value, str)):
                item.body.pop(0)

    class RestoreNames(ast.NodeTransformer):
        def visit_Name(self, item):
            item.id = item.id.replace('annotation', 'gnn').replace('ANNOTATION', 'GNN')
            return item
        def visit_Attribute(self, item):
            self.generic_visit(item)
            item.attr = item.attr.replace('annotation', 'gnn').replace('ANNOTATION', 'GNN')
            return item
        def visit_FunctionDef(self, item):
            item.name = item.name.replace('annotation', 'gnn')
            self.generic_visit(item)
            return item
        def visit_Constant(self, item):
            if isinstance(item.value, str):
                for new, old in [('S1_Gene_Level','S1_Mechanistic'), ('S2_Combined','S2_Mech_Pheno'),
                                 ('feat_annot_','feat_gnn_'), ('has_annot_','has_gnn_'),
                                 ('gene_annotations.csv','final_gnn_dataset.csv'),
                                 ('annotation','gnn'), ('ANNOTATION','GNN')]:
                    item.value = item.value.replace(new, old)
            return item
    normalised = Normalise().visit(RestoreNames().visit(copy.deepcopy(node)))
    return hashlib.sha256(ast.dump(normalised, include_attributes=False).encode()).hexdigest()


def verify_source(package_dir: Path) -> dict:
    package_dir = Path(package_dir)
    baseline = json.loads((reference_directory(package_dir) / 'function_fingerprints.json').read_text())
    tree = ast.parse((package_dir / 'severity_pipeline.py').read_text(encoding='utf-8'))
    current = {n.name.replace('annotation', 'gnn'): n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    changed = []
    missing = []
    identical = []
    allowed = set(baseline['permitted_function_changes'] + baseline['permitted_class_changes'])
    approved = json.loads((reference_directory(package_dir) / 'approved_cleanup_fingerprints.json').read_text())
    for name, item in baseline['definitions'].items():
        if name not in current:
            missing.append(name)
        elif canonical_hash(current[name], baseline['renames']) == item['canonical_sha256']:
            identical.append(name)
        else:
            changed.append(name)
    cleanup_path = reference_directory(package_dir).parent / 'cleanup_source_changes.json'
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.is_file() else {}
    accepted = cleanup.get('benchmark_definitions', {})
    for name in changed:
        if name in accepted and canonical_hash(current[name], baseline['renames']) == accepted[name]:
            allowed.add(name)
    unexpected_changes = sorted(set(changed) - allowed)
    for name in set(changed) & allowed:
        if canonical_hash(current[name], baseline['renames']) != accepted.get(name, approved['definitions'].get(name, {}).get('canonical_sha256')):
            unexpected_changes.append(name)
    for name, expected in approved['new_definitions'].items():
        if name not in current or canonical_hash(current[name], baseline['renames']) != accepted.get(name, expected):
            unexpected_changes.append(name)
    current_globals = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in getattr(node, 'targets', [getattr(node, 'target', None)]):
                if isinstance(target, ast.Name):
                    current_globals[target.id] = canonical_hash(node, baseline['renames'])
    changed_globals = sorted(name for name, expected in approved['preserved_globals'].items()
                             if current_globals.get(name) != cleanup.get('benchmark_globals', {}).get(name, expected))
    return {
        'status': 'passed' if not missing and not unexpected_changes and not changed_globals else 'failed',
        'original_functions': sum(isinstance(n, ast.FunctionDef) for name, n in current.items()
                                  if name in baseline['definitions']),
        'original_classes': sum(isinstance(n, ast.ClassDef) for name, n in current.items()
                                if name in baseline['definitions']),
        'identical_definitions_after_documented_renaming': identical,
        'documented_changes': sorted(set(changed) & allowed),
        'missing_definitions': missing,
        'unexpected_changes': unexpected_changes,
        'preserved_global_assignments': len(approved['preserved_globals']),
        'unexpected_global_changes': changed_globals,
        'new_definitions': sorted(set(current) - set(baseline['definitions'])),
        'cleanup_record': str(cleanup_path) if cleanup else None,
        'note': 'Cleanup hashes record reviewed source edits; they do not establish full-run equivalence. AST comparison ignores comments, docstrings, line positions and documented private renames. It is not a training rerun.',
    }


if __name__ == '__main__':
    result = verify_source(Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
