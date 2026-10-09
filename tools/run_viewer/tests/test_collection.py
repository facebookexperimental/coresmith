"""The collector must include committed WAL data and exclude credential stores."""
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_online_snapshot_preserves_wal_rows_and_excludes_credentials(tmp_path):
    study = tmp_path/'study'
    arm = study/'arms'/'example-treatment'
    engine = arm/'work/.coresmith'
    engine.mkdir(parents=True)
    private = tmp_path/'private'
    native = private/'.claude/projects/demo'
    native.mkdir(parents=True)
    (private/'.claude/.credentials.json').write_text('{"token":"PRIVATE_CREDENTIAL_FIXTURE"}')
    (native/'session.jsonl').write_text('{"type":"user","message":{"content":"visible"}}\n')
    (arm/'config.json').write_text(json.dumps({'condition':'treatment','provider':'claude',
        'model':'example-model','private_home':str(private),'password':'PRIVATE_CREDENTIAL_FIXTURE'}))
    con = sqlite3.connect(engine/'project.sqlite')
    try:
        con.execute('pragma journal_mode=WAL')
        con.execute('create table actions (id integer, command text)')
        con.execute('insert into actions values (1, "coresmith status")')
        con.commit()
        assert (engine/'project.sqlite-wal').stat().st_size > 0
        out = tmp_path/'snapshot'
        script = Path(__file__).resolve().parents[1]/'collect_study.py'
        subprocess.run([sys.executable,str(script),'--study',str(study),'--output',str(out)],
                       check=True,capture_output=True,text=True)
        with sqlite3.connect(out/'arms/example-treatment/work/.coresmith/project.sqlite') as backup:
            assert backup.execute('select count(*) from actions').fetchone()[0] == 1
            assert backup.execute('pragma quick_check').fetchone()[0] == 'ok'
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['errors'] == []
        assert (out/'READY.json').exists()
        for record in manifest['files']:
            assert hashlib.sha256((out/record['path']).read_bytes()).hexdigest() == record['sha256']
        for file in out.rglob('*'):
            if file.is_file():
                assert b'PRIVATE_CREDENTIAL_FIXTURE' not in file.read_bytes()
                assert file.name != '.credentials.json'
        assert con.execute('select count(*) from actions').fetchone()[0] == 1
    finally:
        con.close()
