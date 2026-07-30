# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic-assembler edge binding: compound (slash-named) channel fields
bind cross-named producer/consumer ports; a NAMED port that fails to resolve
is a wiring hazard (LLM-integrator fallback), never a silent skip that ties
the channel to constants."""
from __future__ import annotations

import re

from orchestrator.langgraph.integration_helpers import (
    generate_caravel_wrapper_top,
    parse_verilog_ports,
)


def _wrapper(tmp_path):
    p = tmp_path / "user_project_wrapper.v"
    p.write_text(
        "module user_project_wrapper (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  input wire [37:0] io_in, output wire [37:0] io_out,\n"
        "  output wire [37:0] io_oeb\n"
        ");\n"
        "  assign io_out = 38'b0; assign io_oeb = 38'b0;\n"
        "endmodule\n")
    return str(p)


def _producer(tmp_path):
    p = tmp_path / "regmap.v"
    p.write_text(
        "module regmap (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  output wire m_work_write_srdy,\n"
        "  output wire [7:0] m_work_write_data,\n"
        "  input wire m_work_write_drdy\n"
        ");\n"
        "  assign m_work_write_srdy = 1'b0;\n"
        "  assign m_work_write_data = 8'b0;\n"
        "endmodule\n")
    return str(p)


def _consumer(tmp_path):
    p = tmp_path / "store.v"
    p.write_text(
        "module store (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  input wire s_butterfly_write_srdy,\n"
        "  input wire [7:0] s_butterfly_write_data,\n"
        "  output wire s_butterfly_write_drdy\n"
        ");\n"
        "  assign s_butterfly_write_drdy = 1'b1;\n"
        "endmodule\n")
    return str(p)


def _design(tmp_path, producer_port, consumer_port):
    rtl_paths = {
        "user_project_wrapper": _wrapper(tmp_path),
        "regmap": _producer(tmp_path),
        "store": _consumer(tmp_path),
    }
    modules = {bn: parse_verilog_ports(p) for bn, p in rtl_paths.items()}
    edges = [{
        "producer_block": "regmap", "consumer_block": "store",
        "producer_port": producer_port, "consumer_port": consumer_port,
        "edge_id": "work_write",
    }]
    return modules, edges, rtl_paths


def test_compound_cross_named_channel_binds(tmp_path):
    modules, edges, rtl_paths = _design(
        tmp_path,
        "m_work_write_srdy/m_work_write_data/m_work_write_drdy",
        "s_butterfly_write_srdy/s_butterfly_write_data/s_butterfly_write_drdy",
    )
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert asm["wiring_errors"] == []
    v = asm["verilog"]
    # each field pair shares ONE wire: the wire tied to the producer port also
    # feeds the consumer port -- no constant ties on the channel
    for pf, cf in (("m_work_write_srdy", "s_butterfly_write_srdy"),
                   ("m_work_write_data", "s_butterfly_write_data"),
                   ("m_work_write_drdy", "s_butterfly_write_drdy")):
        mp = re.search(rf"\.{pf}\s*\(\s*(\w+)\s*\)", v)
        mc = re.search(rf"\.{cf}\s*\(\s*(\w+)\s*\)", v)
        assert mp and mc, (pf, cf)
        assert mp.group(1) == mc.group(1), (pf, cf, mp.group(1), mc.group(1))
        assert "1'b0" not in (mp.group(1), mc.group(1))


def test_named_unresolvable_port_is_wiring_error(tmp_path):
    modules, edges, rtl_paths = _design(
        tmp_path, "m_typo_srdy", "s_butterfly_write_srdy")
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert any("does not resolve" in w for w in asm["wiring_errors"])


def test_compound_suffix_mismatch_is_wiring_error(tmp_path):
    modules, edges, rtl_paths = _design(
        tmp_path,
        "m_work_write_srdy/m_work_write_data",
        "s_butterfly_write_data/s_butterfly_write_srdy",  # reversed order
    )
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert any("suffix mismatch" in w for w in asm["wiring_errors"])


def test_unnamed_edge_still_falls_back(tmp_path):
    # no producer_port/consumer_port at all: legacy normalized-key fallback
    # path -- must not become a wiring error
    modules, edges, rtl_paths = _design(tmp_path, "", "")
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert not any("does not resolve" in w for w in asm["wiring_errors"])


class TestBlocksInstantiatedCollision:
    def test_top_named_block_satisfied_by_pads_rename(self):
        from orchestrator.langchain.agents.integration_lead import (
            assert_blocks_instantiated,
        )
        top = (
            "module user_project_wrapper (input wire wb_clk_i);\n"
            "  user_project_wrapper_pads u_user_project_wrapper "
            "(.wb_clk_i(wb_clk_i));\n"
            "  fft_engine u_fft (.clk(wb_clk_i));\n"
            "endmodule\n")
        assert assert_blocks_instantiated(
            top, ["user_project_wrapper", "fft_engine"]) is None

    def test_collision_without_pads_instance_still_caught(self):
        from orchestrator.langchain.agents.integration_lead import (
            assert_blocks_instantiated,
        )
        top = ("module user_project_wrapper (input wire c);\n"
               "  fft_engine u (.clk(c));\nendmodule\n")
        err = assert_blocks_instantiated(
            top, ["user_project_wrapper", "fft_engine"])
        assert err and "user_project_wrapper" in err

    def test_genuinely_missing_block_still_caught(self):
        from orchestrator.langchain.agents.integration_lead import (
            assert_blocks_instantiated,
        )
        top = (
            "module user_project_wrapper (input wire c);\n"
            "  user_project_wrapper_pads u_user_project_wrapper (.c(c));\n"
            "endmodule\n")
        err = assert_blocks_instantiated(
            top, ["user_project_wrapper", "dropped_block"])
        assert err and "dropped_block" in err


# ---------------------------------------------------------------------------
# Channel names: the contract names a CHANNEL, the RTL prefix-expands it.
# ---------------------------------------------------------------------------

def _chan_pair(tmp_path, prod_fields, cons_fields):
    """A producer/consumer pair whose channel `ch` is prefix-expanded."""
    prod = tmp_path / "prod.v"
    prod.write_text(
        "module prod (\n  input wire wb_clk_i, input wire wb_rst_i,\n  "
        + ",\n  ".join(f"output wire {w}ch_{n}" for n, w in prod_fields)
        + "\n);\nendmodule\n")
    cons = tmp_path / "cons.v"
    cons.write_text(
        "module cons (\n  input wire wb_clk_i, input wire wb_rst_i,\n  "
        + ",\n  ".join(f"input wire {w}ch_{n}" for n, w in cons_fields)
        + "\n);\nendmodule\n")
    rtl_paths = {
        "user_project_wrapper": _wrapper(tmp_path),
        "prod": str(prod), "cons": str(cons),
    }
    modules = {bn: parse_verilog_ports(q) for bn, q in rtl_paths.items()}
    edges = [{
        "producer_block": "prod", "consumer_block": "cons",
        "producer_port": "ch", "consumer_port": "ch", "edge_id": "chan",
    }]
    return modules, edges, rtl_paths


class TestChannelNameResolution:
    """An interface contract names a channel; this engine's RTL exposes that
    channel as a group of ports sharing the channel's name as a prefix
    (contract `framebuffer_read` -> `framebuffer_read_req_addr`,
    `framebuffer_read_rdata`, `framebuffer_read_rvalid`, ...).

    Before channel expansion, EVERY such edge failed to resolve, counted as a
    wiring hazard, and took the whole deterministic Caravel assembly down with
    it -- on exp-raster-macro-20260727, all 15 contract edges, so the gradeable
    top was built and then discarded at the last step.
    """

    def test_matching_field_sets_bind_with_no_hazard(self, tmp_path):
        fields = [("addr", "[11:0] "), ("data", "[7:0] "), ("valid", "")]
        modules, edges, rtl_paths = _chan_pair(tmp_path, fields, fields)
        asm = generate_caravel_wrapper_top(
            modules, edges, rtl_paths, str(tmp_path / "out"),
            "user_project_wrapper")
        assert not asm.get("wiring_errors"), asm.get("wiring_errors")
        v = asm["verilog"]
        # All three fields wired, none left dangling on a constant.
        for n in ("addr", "data", "valid"):
            assert f"ch_{n}" in v

    def test_an_exact_port_name_still_wins(self, tmp_path):
        """A block that really HAS a port of that name must be unaffected --
        channel expansion is a fallback, not a replacement."""
        modules, edges, rtl_paths = _design(
            tmp_path, "m_work_write_srdy", "s_butterfly_write_srdy")
        asm = generate_caravel_wrapper_top(
            modules, edges, rtl_paths, str(tmp_path / "out"),
            "user_project_wrapper")
        assert not asm.get("wiring_errors"), asm.get("wiring_errors")

    def test_a_channel_that_matches_nothing_is_still_a_hazard(self, tmp_path):
        fields = [("addr", "[11:0] ")]
        modules, edges, rtl_paths = _chan_pair(tmp_path, fields, fields)
        edges[0]["producer_port"] = "no_such_channel"
        asm = generate_caravel_wrapper_top(
            modules, edges, rtl_paths, str(tmp_path / "out"),
            "user_project_wrapper")
        errs = asm.get("wiring_errors") or []
        assert errs and any("does not resolve" in e for e in errs)

    def test_asymmetric_field_sets_are_refused_not_guessed(self, tmp_path):
        """The REMAINING blocker, pinned deliberately.

        The two sides of a channel often expose different field names
        (aperture `host_write_enable` vs store `host_write_accept`). Positional
        pairing would then cross fields, so the assembler refuses -- correctly.
        Closing this needs suffix-KEYED matching with handshake aliasing, which
        is a design decision, not a parser tweak. Until then a design with
        asymmetric channel fields still falls back to the LLM integrator.
        """
        modules, edges, rtl_paths = _chan_pair(
            tmp_path,
            [("enable", ""), ("addr", "[11:0] ")],
            [("accept", ""), ("addr", "[11:0] ")])
        asm = generate_caravel_wrapper_top(
            modules, edges, rtl_paths, str(tmp_path / "out"),
            "user_project_wrapper")
        errs = asm.get("wiring_errors") or []
        assert errs, "crossed two differently-named channel fields"
        assert any("suffix mismatch" in e for e in errs), errs

    def test_width_mismatch_across_a_channel_is_refused(self, tmp_path):
        modules, edges, rtl_paths = _chan_pair(
            tmp_path, [("data", "[7:0] ")], [("data", "[15:0] ")])
        asm = generate_caravel_wrapper_top(
            modules, edges, rtl_paths, str(tmp_path / "out"),
            "user_project_wrapper")
        errs = asm.get("wiring_errors") or []
        assert any("width" in e for e in errs), errs
