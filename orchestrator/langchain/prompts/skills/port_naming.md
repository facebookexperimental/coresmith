# Skill: canonical port naming (`<channel>_<field>`)

The interface contract is the SOLE naming authority for a block's ports.

## The rule

For every edge that touches this block, the contract names a channel (the
block's `producer_port` / `consumer_port`) and lists the signals on it
(`fields[]` + `sideband_signals[]` -- their UNION is the port set). The
canonical flattened RTL port name is:

    <channel>_<field>

A signal the contract lists with no channel (a bare sideband such as `clk`,
`rst_n`, or an externally mandated pad name) keeps its bare name.

## NEVER shorten a doubled token

When the channel suffix and the field prefix share a token, the token appears
TWICE. That is correct and required:

    channel: data_write        field: write_enable
    canonical port name:  data_write_write_enable      <-- CORRECT
    WRONG:                data_write_enable            <-- collapsed token
    WRONG:                write_enable                 <-- dropped channel

The doubled token is not a typo and not redundant. The deterministic
integration assembler resolves contract edges BY NAME: `data_write_enable`
does not resolve, the edge is unwireable, and the block is rejected by the
pre-simulation conformance gate. Two channels on one block routinely carry
the same field name (`read_enable`, `req_addr`, `valid`), and the channel
prefix is the only thing that tells them apart.

Same rule for a dropped or substituted prefix: `host_read_enable` is not an
acceptable spelling of `framebuffer_read_read_enable`.

## Precedence

1. **The contract's port table wins.** It is the frozen, authoritative naming
   source.
2. A golden / reference model's port identifiers are NOT authoritative. Models
   are written for simulation and routinely collapse or abbreviate names.
   Transcribe the model's BEHAVIOR byte-exact; take every port NAME from the
   contract table.
3. Accumulated per-block constraints, prior attempts' RTL, testbench code, and
   the uArch spec's prose are all subordinate to the contract table on naming.
   If any of them names a port differently, the contract wins -- silently and
   without asking.

## Checklist before emitting a module header

- [ ] Every contract signal on every edge touching this block has a port.
- [ ] Each port is spelled `<channel>_<field>` exactly, character for
      character, including any doubled token.
- [ ] No port wears a channel prefix that the contract does not declare.
- [ ] No signal is exposed twice (both `<channel>_<field>` and bare `<field>`).
