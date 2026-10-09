"""Read-only study evidence capture. Run on the worker; never execute run code."""
import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', action='append', help='select an arm; repeat to select several (default: treatment arms)')
    args = ap.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at=now(), study=str(study), files=[], errors=[],
                    consistency='Each SQLite file is an online backup; logs are bounded copies. The cross-file snapshot is not atomic.',
                    scope='Selected CoreSmith arms: CLI audit and all engine SQLite databases, active/rotated graph and LLM logs, text step logs, all native sessions and Architect invocation transcripts. Credential/configuration stores, RTL, binary build artifacts and unrelated study arms are excluded.')

    def destination(relative):
        p = out / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def record_error(src, exc):
        manifest['errors'].append(dict(source=str(src), error=str(exc)))

    def copy(src, relative):
        if src.is_symlink() or not src.is_file():
            return
        try:
            before = src.stat()
            target = destination(relative)
            h = hashlib.sha256()
            remain = before.st_size
            last = b''
            with src.open('rb') as source, target.open('wb') as dest:
                while remain:
                    chunk = source.read(min(remain, 1024 * 1024))
                    if not chunk:
                        break
                    dest.write(chunk)
                    h.update(chunk)
                    last = chunk[-1:]
                    remain -= len(chunk)
            after = src.stat()
            manifest['files'].append(dict(path=str(relative), source=str(src), mode='bounded-copy',
                bytes=target.stat().st_size, source_bytes_before=before.st_size,
                source_bytes_after=after.st_size, source_mtime_ns=before.st_mtime_ns,
                changed_during_copy=(before.st_size, before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns),
                partial_final_line=(src.suffix=='.jsonl' and bool(last) and last!=b'\n'),
                sha256=h.hexdigest()))
        except (OSError, ValueError) as exc:
            record_error(src, exc)

    def backup(src, relative):
        if src.is_symlink():
            return
        start = time.monotonic()
        target = destination(relative)
        def progress(status, remaining, total):
            if time.monotonic()-start > 45:
                raise TimeoutError('SQLite backup exceeded 45 seconds')
        try:
            with sqlite3.connect(src.as_uri()+'?mode=ro', uri=True, timeout=10) as source:
                with sqlite3.connect(target) as dest:
                    source.backup(dest, pages=256, progress=progress, sleep=0.05)
            h = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest['files'].append(dict(path=str(relative), source=str(src), mode='sqlite-online-backup',
                bytes=target.stat().st_size, sha256=h, elapsed_seconds=round(time.monotonic()-start, 3)))
        except (OSError, sqlite3.Error, TimeoutError) as exc:
            if target.exists():
                target.unlink()
            record_error(src, exc)

    selected = args.arm
    if selected is None:
        selected = [p.parent.name for p in sorted((study/'arms').glob('*/config.json'))
                    if json.loads(p.read_text()).get('condition') == 'treatment']
    if not selected:
        raise SystemExit('No treatment arms found; select an arm with --arm')
    for arm in selected:
        if Path(arm).name != arm or arm in ('.', '..'):
            raise SystemExit('Arm names must name direct children of the study arms directory')
        srcarm = study/'arms'/arm
        relarm = Path('arms')/arm
        cfg = json.loads((srcarm/'config.json').read_text())
        public_cfg = {k:cfg[k] for k in ('run','arm','condition','provider','model','effort',
            'container','engine_commit','engine_tree_sha256','image') if k in cfg}
        destination(relarm/'config.json').write_text(json.dumps(public_cfg, indent=2)+'\n')
        for name in ('status.json', 'initial-prompt.txt', 'interventions.jsonl'):
            copy(srcarm/name, relarm/name)
        for inv in sorted((srcarm/'invocations').glob('*')):
            if inv.is_dir():
                for name in ('transcript.jsonl','status.json','command.json','prompt.txt','stderr.log',
                             'response.txt','result.json'):
                    copy(inv/name, relarm/'invocations'/inv.name/name)
        home = Path(cfg['private_home'])
        for relative in ('.claude/projects', '.codex/sessions'):
            root = home/relative
            if root.exists():
                for src in sorted(root.rglob('*.jsonl')):
                    copy(src, relarm/'native-sessions'/src.relative_to(home))
        engine = srcarm/'work/.coresmith'
        for src in sorted(engine.iterdir()):
            relative = relarm/'work/.coresmith'/src.name
            if src.suffix in ('.sqlite','.db'):
                backup(src, relative)
            elif src.is_file() and src.suffix in ('.json','.jsonl','.log','.txt','.md'):
                copy(src, relative)
            elif src.is_dir() and src.name in ('step_logs','live_streams','logs'):
                for nested in sorted(src.rglob('*')):
                    if nested.is_file() and not nested.is_symlink():
                        copy(nested, relarm/'work/.coresmith'/nested.relative_to(engine))
        print(json.dumps(dict(arm=arm, files=len(manifest['files']), errors=len(manifest['errors']))), flush=True)
    for name in ('execution.json','manifest.json'):
        copy(study/name, Path('study')/name)
    manifest['finished_at'] = now()
    manifest['bytes'] = sum(r['bytes'] for r in manifest['files'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'READY.json').write_text(json.dumps(dict(started_at=manifest['started_at'],
        finished_at=manifest['finished_at'], files=len(manifest['files']), bytes=manifest['bytes'],
        errors=len(manifest['errors'])), indent=2)+'\n')
    print((out/'READY.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
