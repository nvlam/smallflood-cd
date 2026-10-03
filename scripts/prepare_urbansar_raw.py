"""Verify the official archive, safely extract it, and recover test TIFFs from LFS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(
            hashlib, 'file_digest'
        ) else stream_digest(handle)


def stream_digest(handle):
    result = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def pointer(root, relative):
    value = subprocess.check_output(
        ['git', '-C', str(root), 'show', f'HEAD:{relative}'], text=True
    )
    fields = dict(line.split(' ', 1) for line in value.splitlines())
    return fields['oid'].split(':')[1], int(fields['size'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / 'urban_sar_floods.tar.gz'
    sha, size = pointer(root, archive.name)
    if archive.stat().st_size != size or digest(archive) != sha:
        raise RuntimeError('Archive checksum mismatch')
    print('Archive SHA-256 verified', sha, flush=True)
    extracted = root / 'extracted_v1'
    marker = extracted / 'extraction_complete.json'
    if not marker.exists():
        extracted.mkdir(exist_ok=True)
        count = 0
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted) or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f'Unsafe member: {member.name}')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(f'Incomplete extraction: {target}')
                    with source.extractfile(member) as reader, target.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
                    count += 1
                    if count % 1000 == 0:
                        print('Extracted files', count, flush=True)
        marker.write_text(json.dumps({'archive_sha256': sha, 'files': count}, indent=2))
        print('Extraction complete', count, flush=True)
    for path in sorted((root / 'testing_case_orig').glob('*/*.tif')):
        sha, size = pointer(root, str(path.relative_to(root)))
        if path.stat().st_size == size and digest(path) == sha:
            print('Test TIFF verified', path.name, flush=True)
            continue
        if path.stat().st_size != 0:
            raise RuntimeError(f'Refusing to replace nonempty invalid file: {path}')
        cached = root / '.git/lfs/objects' / sha[:2] / sha[2:4] / sha
        if not cached.exists() or cached.stat().st_size != size or digest(cached) != sha:
            raise RuntimeError(f'Test TIFF missing or invalid in cache: {path}')
        temporary = path.with_suffix('.verified.tmp')
        with cached.open('rb') as reader, temporary.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        temporary.replace(path)
        print('Test TIFF restored', path.name, size, flush=True)


if __name__ == '__main__':
    main()
