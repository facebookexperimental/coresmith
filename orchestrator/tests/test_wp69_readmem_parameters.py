"""Parameterised memory assets through adoption, validation and Yosys."""
import shutil
from pathlib import Path

import pytest

from orchestrator.harness import top_module as tm


def design(root, override='', default='""', *, body_parameter=False, readmem='INIT_FILE'):
    (root / 'inputs').mkdir(exist_ok=True)
    (root / 'rtl').mkdir(exist_ok=True)
    top = root / 'rtl/top.v'
    memory = root / 'rtl/memory.v'
    top.write_text('module chip_top(output [7:0] y); parameter OTHER="rom.hex";\n'
                   f'memory {override} u_rom(.y(y)); endmodule\n')
    declaration = f'parameter WIDTH=8, INIT_FILE={default}'
    header = '(output [7:0] y);\n' + declaration + ';' if body_parameter else (
        f'#({declaration})(output [7:0] y);')
    memory.write_text(f'module memory {header}\nreg [7:0] mem[0:0];\n'
                      f'initial if (INIT_FILE != "") $readmemh({readmem}, mem);\n'
                      'assign y=mem[0]; endmodule\n')
    return top, memory


def adopt(root, top, memory):
    return tm.write_candidate_receipt(root, 'chip_top', str(top), {'memory': str(memory)})


@pytest.fixture
def tool_free(monkeypatch):
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    monkeypatch.setattr('orchestrator.harness.hierarchy.elaborate_hierarchy',
                        lambda *a, **k: {'memory'})


def test_empty_default_adopts_without_asset(tmp_path, tool_free):
    top, memory = design(tmp_path)
    rec = adopt(tmp_path, top, memory)
    assert rec['dependencies'] == []
    assert tm.validated_candidate(tmp_path)['candidate_sha'] == rec['candidate_sha']


@pytest.mark.parametrize('override', ['#(.INIT_FILE("rom.hex"))', '#(8, "rom.hex")'])
@pytest.mark.parametrize('body_parameter', [False, True])
def test_literal_override_is_bound_and_mutation_stales(tmp_path, tool_free, override, body_parameter):
    top, memory = design(tmp_path, override, body_parameter=body_parameter)
    asset = tmp_path / 'inputs/rom.hex'
    asset.write_text('12\n')
    rec = adopt(tmp_path, top, memory)
    assert rec['dependencies'] == [str(asset)]
    asset.write_text('34\n')
    assert not tm.receipt_is_current(rec)
    with pytest.raises(tm.CandidateError, match='stale'):
        tm.validated_candidate(tmp_path)
    assert adopt(tmp_path, top, memory)['candidate_sha'] != rec['candidate_sha']


def test_all_defaults_and_instance_literals_are_bound(tmp_path, tool_free):
    top, memory = design(tmp_path, '#(.INIT_FILE("first.hex"))', '"default.hex"')
    top.write_text(top.read_text().replace('endmodule',
        'memory #(8, "second.hex") u_second(); endmodule'))
    assets = [tmp_path / 'inputs' / name for name in ('default.hex', 'first.hex', 'second.hex')]
    for asset in assets:
        asset.write_text('12\n')
    assert adopt(tmp_path, top, memory)['dependencies'] == sorted(map(str, assets))


@pytest.mark.parametrize('override', [
    '#(.INIT_FILE(OTHER))', '#(.INIT_FILE({"rom", ".hex"}))',
    '#(8, OTHER)', '#(8, {"rom", ".hex"})', '#(.INIT_FILE(`IMAGE))',
    '#(.INIT_FILE("rom.hex"), .INIT_FILE("rom.hex"))', '#(8, "rom.hex", 1)',
    '#(.INIT_FILE())',
])
def test_unresolved_override_names_site_and_leaves_no_receipt(tmp_path, tool_free, override):
    top, memory = design(tmp_path, override)
    (tmp_path / 'inputs/rom.hex').write_text('12\n')
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    for name in ('memory', 'INIT_FILE', 'u_rom'):
        assert name in str(error.value)
    assert not (tmp_path / tm.RECEIPT_REL).exists()


@pytest.mark.parametrize('default', ['OTHER', '{"rom", ".hex"}'])
def test_nonliteral_default_names_parameter(tmp_path, tool_free, default):
    top, memory = design(tmp_path, default=default)
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert 'memory' in str(error.value) and 'INIT_FILE' in str(error.value)


def test_defparam_cannot_escape_binding(tmp_path, tool_free):
    top, memory = design(tmp_path)
    top.write_text(top.read_text().replace('endmodule',
        'defparam u_rom.INIT_FILE = "rom.hex"; endmodule'))
    (tmp_path / 'inputs/rom.hex').write_text('12\n')
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert all(name in str(error.value) for name in ('memory', 'INIT_FILE', 'u_rom'))


def test_literal_paths_preserve_comment_characters_and_commas(tmp_path, tool_free):
    top, memory = design(tmp_path, '#(.INIT_FILE("images//rom,1.hex"))')
    asset = tmp_path / 'inputs/images/rom,1.hex'
    asset.parent.mkdir()
    asset.write_text('12\n')
    assert adopt(tmp_path, top, memory)['dependencies'] == [str(asset)]


@pytest.mark.parametrize('location', ['missing', 'ambiguous'])
def test_parameter_asset_uses_literal_resolution_rules(tmp_path, tool_free, location):
    top, memory = design(tmp_path, '#(.INIT_FILE("rom.hex"))')
    if location == 'ambiguous':
        (tmp_path / 'inputs/rom.hex').write_text('12\n')
        (top.parent / 'rom.hex').write_text('34\n')
    with pytest.raises(tm.CandidateError, match=location):
        adopt(tmp_path, top, memory)


@pytest.mark.parametrize('override,default', [
    ('', '""'), ('#(.INIT_FILE("rom.hex"))', '""'), ('#(8, "rom.hex")', '""'),
    ('', '"rom.hex"'),
])
def test_parameter_assets_elaborate_with_real_yosys(tmp_path, monkeypatch, override, default):
    if not shutil.which('yosys'):
        pytest.skip('requires yosys')
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    top, memory = design(tmp_path, override, default)
    (tmp_path / 'inputs/rom.hex').write_text('a5\n')
    rec = adopt(tmp_path, top, memory)
    assert 'memory' in rec['elaborated_cells']
    assert tm.receipt_is_current(rec)


def test_engine_flop_memory_candidate_adopts(tmp_path, monkeypatch):
    if not shutil.which('yosys'):
        pytest.skip('requires yosys')
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    top = tmp_path / 'top.v'
    top.write_text('module chip_top(input clk, output [7:0] y);\n'
        'cs_fpmem_1rw1r #(.WIDTH(8), .DEPTH(2)) u_memory(.clk(clk), .ce0(1\'b1), '
        '.we0(1\'b0), .addr0(1\'b0), .wdata0(8\'b0), .ce1(1\'b0), '
        '.addr1(1\'b0), .rdata0(y)); endmodule\n')
    rec = tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})
    assert 'cs_fpmem_1rw1r' in rec['elaborated_cells']
    assert rec['dependencies'] == []
    assert all(Path(p).is_file() for p in rec['sources'])


def test_top_parameter_override_cannot_bypass_asset_binding(tmp_path, tool_free):
    top, memory = design(tmp_path)
    with pytest.raises(tm.CandidateError) as error:
        tm.write_candidate_receipt(tmp_path, 'memory', str(memory), {},
                                   parameters={'INIT_FILE': 123}, expected_blocks=[])
    assert 'memory' in str(error.value) and 'INIT_FILE' in str(error.value)


def test_readmem_in_include_keeps_enclosing_module(tmp_path, tool_free):
    top, memory = design(tmp_path, '#(.INIT_FILE("rom.hex"))')
    include = tmp_path / 'inputs/init.vh'
    include.write_text('initial if (INIT_FILE != "") $readmemh(INIT_FILE,mem);\n')
    memory.write_text(memory.read_text().replace(
        'initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);', '`include "init.vh"'))
    asset = tmp_path / 'inputs/rom.hex'
    asset.write_text('12\n')
    assert adopt(tmp_path, top, memory)['dependencies'] == sorted([str(asset), str(include)])


def test_library_selection_is_shared_and_unused_edits_do_not_change_sha(tmp_path, monkeypatch):
    import asyncio

    from orchestrator.langgraph import backend_graph as bg
    from orchestrator.langgraph import integration_helpers as ih
    from orchestrator.langgraph import sram_wrapper

    library = tmp_path / 'library.v'
    library.write_text('module cs_fpmem_1rw1r(); endmodule\n'
        'module unused #(parameter INIT_FILE=""); reg mem[0:0];\n'
        'initial if (INIT_FILE != "") $readmemh(INIT_FILE,mem); endmodule\n'
        'module unused_wrapper #(parameter INIT_FILE="");\n'
        'unused #(.INIT_FILE(INIT_FILE)) unresolved(); endmodule\n')
    top = tmp_path / 'top.v'
    top.write_text('module chip_top(); cs_fpmem_1rw1r u(); endmodule\n')
    monkeypatch.setattr(sram_wrapper, 'wrapper_lib_path', lambda: str(library))
    monkeypatch.setattr('orchestrator.harness.hierarchy.elaborate_hierarchy',
                        lambda *a, **k: {'cs_fpmem_1rw1r'})
    rec = tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})
    from orchestrator.harness.task_adapter import assemble_candidate
    assert assemble_candidate(str(tmp_path), str(top), {})['sources'] == rec['sources']
    assert ih.chip_rtl_sources(str(top), {}, top_module='chip_top',
                              project_root=tmp_path) == rec['sources']
    backend = asyncio.run(bg.init_design_node({'project_root': str(tmp_path)}))
    assert {backend['integration_top_path'], *backend['block_rtl_paths'].values()} == set(rec['sources'])
    assert str(library) not in rec['sources']
    library.write_text(library.read_text().replace('module unused #', 'module unused_changed #'))
    assert tm.receipt_is_current(rec)
    assert tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})['candidate_sha'] == rec['candidate_sha']
    # Selected engine text changes affect the next adopted candidate; previous
    # manifests retain their immutable source snapshot.
    library.write_text(library.read_text().replace('1rw1r();', '1rw1r(); wire changed;'))
    assert tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})['candidate_sha'] != rec['candidate_sha']


def test_escaped_instance_override_is_bound(tmp_path, tool_free):
    top, memory = design(tmp_path, '#(.INIT_FILE("rom.hex"))')
    top.write_text(top.read_text().replace('u_rom(', '\\u-rom ('))
    asset = tmp_path / 'inputs/rom.hex'
    asset.write_text('12\n')
    assert adopt(tmp_path, top, memory)['dependencies'] == [str(asset)]


def test_macro_instance_cannot_hide_nonliteral_override(tmp_path, tool_free):
    top, memory = design(tmp_path, '#(.INIT_FILE(OTHER))')
    top.write_text('`define MEM memory\n' + top.read_text().replace('memory #', '`MEM #'))
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert all(name in str(error.value) for name in ('memory', 'INIT_FILE', 'u_rom'))


def test_ancestor_defparam_cannot_hide_override(tmp_path, tool_free):
    top, memory = design(tmp_path)
    top.write_text('module chip_top(output [7:0] y); wrapper u_wrapper(y);\n'
        'defparam u_wrapper.u_rom.INIT_FILE="rom.hex"; endmodule\n'
        'module wrapper(output [7:0] y); memory u_rom(y); endmodule\n')
    (tmp_path / 'inputs/rom.hex').write_text('12\n')
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert all(name in str(error.value) for name in ('memory', 'INIT_FILE', 'u_rom'))


def test_unguarded_empty_default_is_no_asset_in_yosys(tmp_path, monkeypatch):
    if not shutil.which('yosys'):
        pytest.skip('requires yosys')
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    top, memory = design(tmp_path)
    memory.write_text(memory.read_text().replace('if (INIT_FILE != "") ', ''))
    rec = adopt(tmp_path, top, memory)
    assert rec['dependencies'] == []


def test_uninstantiated_library_branch_is_absent_from_manifest(tmp_path, monkeypatch):
    from orchestrator.langgraph import sram_wrapper
    library = tmp_path / 'library.v'
    library.write_text('module cs_fpmem_1rw1r();\n'
        'if (0) begin:dead unused u(); end endmodule\nmodule unused(); endmodule\n')
    top = tmp_path / 'top.v'
    top.write_text('module chip_top(); cs_fpmem_1rw1r u(); endmodule\n')
    monkeypatch.setattr(sram_wrapper, 'wrapper_lib_path', lambda: str(library))
    monkeypatch.setattr('orchestrator.harness.hierarchy.elaborate_hierarchy',
                        lambda *a, **k: {'cs_fpmem_1rw1r'})
    rec = tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})
    assert len(rec['sources']) == 2
    library.write_text(library.read_text().replace('unused();', 'unused(); wire changed;'))
    assert tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})['candidate_sha'] == rec['candidate_sha']


@pytest.mark.parametrize('header', ['', '#(parameter INIT_FILE="")'])
def test_generate_local_parameter_is_not_a_module_asset_parameter(tmp_path, tool_free, header):
    top, memory = design(tmp_path)
    memory.write_text(f'module memory {header}(output [7:0] y);\n'
        'if (1) begin:local_scope parameter INIT_FILE=""; reg [7:0] mem[0:0];\n'
        'initial $readmemh(INIT_FILE,mem); assign y=mem[0]; end endmodule\n')
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert 'memory' in str(error.value) and 'INIT_FILE' in str(error.value)


def engine_wrapper_design(root, override=''):
    (root / 'inputs').mkdir(exist_ok=True)
    top = root / 'top.v'
    top.write_text('module chip_top(input clk, output [7:0] y); parameter OTHER="rom.hex";\n'
        f'cs_sram_1rw1r #(.WIDTH(8), .DEPTH(2){override}) u_mem(.clk(clk), .ce0(1\'b0), '
        '.we0(1\'b0), .addr0(1\'b0), .wdata0(8\'b0), .wmask0(1\'b0), .rdata0(y), .ce1(1\'b0), '
        '.addr1(1\'b0), .rdata1()); endmodule\n')
    return top


def test_engine_wrapper_forwards_init_file_without_binding_an_asset(tmp_path, monkeypatch):
    # Q26(a): the engine's cs_sram_1rw1r forwards .INIT_FILE(INIT_FILE) to
    # cs_mem_1rw1r. That engine text is not a candidate asset; with no
    # candidate init file nothing is bound and adoption succeeds.
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    monkeypatch.setattr('orchestrator.harness.hierarchy.elaborate_hierarchy',
                        lambda *a, **k: {'cs_sram_1rw1r', 'cs_mem_1rw1r'})
    rec = tm.write_candidate_receipt(tmp_path, 'chip_top', str(engine_wrapper_design(tmp_path)), {})
    assert rec['dependencies'] == []
    assert any('/.coresmith/candidate-library/cs_sram_1rw1r-' in p for p in rec['sources'])
    assert tm.validated_candidate(tmp_path)['candidate_sha'] == rec['candidate_sha']


def test_engine_wrapper_still_binds_or_rejects_candidate_init_files(tmp_path, monkeypatch):
    monkeypatch.delenv('CORESMITH_TOP_MODULE', raising=False)
    monkeypatch.setattr('orchestrator.harness.hierarchy.elaborate_hierarchy',
                        lambda *a, **k: {'cs_sram_1rw1r', 'cs_mem_1rw1r'})
    asset = tmp_path / 'inputs/rom.hex'
    top = engine_wrapper_design(tmp_path, ', .INIT_FILE("rom.hex")')
    asset.write_text('12\n')
    rec = tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})
    assert rec['dependencies'] == [str(asset)]
    asset.write_text('34\n')
    assert not tm.receipt_is_current(rec)
    engine_wrapper_design(tmp_path, ', .INIT_FILE(OTHER)')
    with pytest.raises(tm.CandidateError) as error:
        tm.write_candidate_receipt(tmp_path, 'chip_top', str(top), {})
    assert all(name in str(error.value) for name in ('cs_sram_1rw1r', 'INIT_FILE', 'u_mem'))


def test_candidate_authored_forward_stays_unresolved(tmp_path, tool_free):
    # Only engine-written wrapper text may forward; the same shape in candidate
    # RTL is still an unresolved override (as '#(.INIT_FILE(OTHER))' above).
    top, memory = design(tmp_path)
    top.write_text('module chip_top(output [7:0] y); wrapper u_wrapper(y); endmodule\n'
        'module wrapper #(parameter INIT_FILE="")(output [7:0] y);\n'
        'memory #(.INIT_FILE(INIT_FILE)) u_rom(.y(y)); endmodule\n')
    with pytest.raises(tm.CandidateError) as error:
        adopt(tmp_path, top, memory)
    assert all(name in str(error.value) for name in ('memory', 'INIT_FILE', 'u_rom'))
