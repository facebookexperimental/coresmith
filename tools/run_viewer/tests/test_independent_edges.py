"""Independent checks for provenance and privacy failures in a run viewer."""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.run_viewer.cli_calls import parse_invocations
from tools.run_viewer.engine import load_events
from tools.run_viewer.native import Agent, claude_record, parse_codex_rollout
from tools.run_viewer.privacy import Ledger, redact
from tools.run_viewer.sources import Sources, parse_iso


def test_command_names_used_as_arguments_are_not_cli_invocations():
    for command in ['echo coresmith build module cpu', 'command -v coresmith',
                    'which coresmith', 'rg coresmith README.md',
                    'printf "%s\\n" coresmith', 'cat /tmp/log | grep coresmith']:
        assert parse_invocations(command)['invocations'] == [], command
    assert parse_invocations('coresmith build module cpu')['invocations'][0]['argv'] == ['build','module','cpu']


def test_unexecuted_shell_branches_are_not_completed_cli_calls():
    # Static source alone cannot prove that a command in a conditional ran.
    # It can be shown as an invocation candidate; audit evidence decides execution.
    result = parse_invocations('false && coresmith build module cpu')
    for candidate in result['invocations']:
        assert candidate.get('execution') != 'confirmed'


def test_codex_analysis_message_is_withheld(tmp_path):
    token='SYNTHETIC_PRIVATE_ANALYSIS_VALUE_1234567890'
    path=tmp_path/'rollout.jsonl'
    path.write_text(json.dumps({'type':'response_item','timestamp':'2026-10-08T12:00:00Z',
        'payload':{'type':'message','id':'secret-message','role':'assistant','channel':'analysis',
                   'content':[{'type':'output_text','text':token}]}})+'\n')
    agent=Agent('test','a',kind='architect',label='Architect',provider='codex')
    parse_codex_rollout(agent,path,Sources(tmp_path),Ledger(),'native')
    assert token not in json.dumps(agent.turns)


def test_nested_claude_private_content_is_withheld():
    token='SYNTHETIC_PRIVATE_THINKING_VALUE_1234567890'
    agent=Agent('test','a',kind='architect',label='Architect',provider='claude')
    event={'type':'user','uuid':'nested','message':{'content':[{'type':'tool_result',
        'tool_use_id':'t','content':[{'type':'thinking','thinking':token,'signature':'synthetic-signature'},
                                  {'type':'text','text':'public tool output'}]}]}}
    claude_record(agent,event,{'s':'S1','l':1},Ledger())
    text=json.dumps(agent.turns)
    assert token not in text and 'synthetic-signature' not in text
    assert 'public tool output' in text


def test_repeated_user_prompts_are_preserved_across_turns(tmp_path):
    path=tmp_path/'rollout.jsonl'
    records=[]
    for n in (1,2):
        records += [
            {'type':'event_msg','timestamp':f'2026-10-08T12:0{n}:00Z',
             'payload':{'type':'task_started','turn_id':f'turn{n}'}},
            {'type':'response_item','timestamp':f'2026-10-08T12:0{n}:01Z',
             'payload':{'type':'message','id':None,'role':'user',
                        'content':[{'type':'input_text','text':'continue'}]}},
            {'type':'event_msg','timestamp':f'2026-10-08T12:0{n}:01Z',
             'payload':{'type':'item_completed','turn_id':f'turn{n}',
                        'item':{'type':'UserMessage','id':f'user{n}',
                                'content':[{'type':'input_text','text':'continue'}]}}}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    agent=Agent('test','a',kind='architect',label='Architect',provider='codex')
    parse_codex_rollout(agent,path,Sources(tmp_path),Ledger(),'native')
    assert len([t for t in agent.turns if t['kind']=='user' and t['text']=='continue'])==2


def test_rotated_copy_does_not_erase_within_file_repeats(tmp_path):
    event={'ts':1.0,'event':'graph_node_enter','node':'test','pid':1}
    line=json.dumps(event)+'\n'
    (tmp_path/'pipeline_events.20261008-120000.jsonl').write_text(line)
    (tmp_path/'pipeline_events.jsonl').write_text(line+line)
    result=load_events(tmp_path,Sources(tmp_path),'test',Ledger())
    assert len(result['events'])==2


def test_naive_engine_timestamps_use_utc():
    old=os.environ.get('TZ')
    try:
        os.environ['TZ']='America/Los_Angeles'
        time.tzset()
        assert parse_iso('2026-10-08T12:00:00')==datetime(2026,10,8,12,tzinfo=timezone.utc).timestamp()
    finally:
        if old is None:
            os.environ.pop('TZ',None)
        else:
            os.environ['TZ']=old
        time.tzset()


def test_private_fields_inside_encoded_tool_output_are_withheld():
    secret = 'SYNTHETIC_PRIVATE_SIGNATURE_' + 'Ab9+' * 40
    thought = 'SYNTHETIC_PRIVATE_THOUGHT_' + 'hidden content ' * 10
    value = {'type':'thinking', 'thinking':thought, 'signature':secret,
             'visible_output':'preserve this visible output'}
    for depth in range(1, 4):
        text = value
        for _ in range(depth):
            text = json.dumps(text)
        result = redact('tool log prefix\n' + text + '\nend of tool log', Ledger())
        assert secret not in result and thought not in result
        assert 'preserve this visible output' in result


def test_snapshot_nested_under_export_output_is_rejected(tmp_path):
    from tools.run_viewer.export import export
    snapshot = tmp_path/'data'/'snapshot'
    snapshot.mkdir(parents=True)
    marker = snapshot/'preserve.txt'
    marker.write_text('original evidence')
    try:
        export(snapshot, tmp_path)
    except SystemExit:
        pass
    else:
        raise AssertionError('overlapping output and snapshot paths must be rejected')
    assert marker.read_text() == 'original evidence'
