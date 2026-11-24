"""Check poso library structure and source snapshots; no network or writes."""
import argparse
import hashlib
import json
from pathlib import Path


def check(root):
    root = root.resolve()
    errors = []

    def local(value, owner):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{owner}: missing relative path")
        target = (root / value).resolve()
        if Path(value).is_absolute() or not target.is_relative_to(root):
            raise ValueError(f"{owner}: path escapes skill: {value}")
        if not target.is_file():
            raise ValueError(f"{owner}: missing file: {value}")
        return target

    index = json.loads(local('styles/index.json', 'index').read_text(encoding='utf-8'))
    manifest = json.loads(local('styles/sources/manifest.json', 'sources').read_text(encoding='utf-8'))
    if index.get('schema_version') != 1 or not index.get('catalog_version'):
        errors.append('index: unsupported schema or absent catalog version')
    sources = manifest.get('sources', [])
    source_ids = set()
    for source in sources:
        owner = source.get('id', '<missing source id>')
        try:
            if owner == '<missing source id>' or owner in source_ids:
                raise ValueError(f'{owner}: missing or duplicate source id')
            source_ids.add(owner)
            for field in ('url', 'revision', 'retrieved_at', 'license_id', 'sha256'):
                if not source.get(field):
                    raise ValueError(f'{owner}: missing {field}')
            snapshot = local(source.get('snapshot'), owner)
            local(source.get('license_file'), owner)
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            if digest != source['sha256']:
                raise ValueError(f'{owner}: snapshot hash mismatch')
        except ValueError as exc:
            errors.append(str(exc))
    seen = set()
    cards = index.get('cards', [])
    if not cards:
        errors.append('index: no cards')
    for card in cards:
        owner = card.get('id', '<missing card id>')
        try:
            if owner == '<missing card id>' or owner in seen:
                raise ValueError(f'{owner}: missing or duplicate card id')
            seen.add(owner)
            for field in ('name', 'version', 'tags', 'fit', 'avoid'):
                if not card.get(field):
                    raise ValueError(f'{owner}: missing {field}')
            if card.get('availability') not in ('active', 'paused'):
                raise ValueError(f'{owner}: invalid availability')
            if card.get('source_id') not in source_ids:
                raise ValueError(f'{owner}: source not found')
            local(card.get('card'), owner)
            if card.get('preview_local'):
                local(card['preview_local'], owner)
        except ValueError as exc:
            errors.append(str(exc))
    return errors, len(cards), len(sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        problems, card_count, source_count = check(args.root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Library invalid: {exc}\n')
    if problems:
        parser.exit(1, '\n'.join(problems) + '\n')
    print(f'Library OK: {card_count} cards, {source_count} pinned sources. Visual results and rights are not certified.')
