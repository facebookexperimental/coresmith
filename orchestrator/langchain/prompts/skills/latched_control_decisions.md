# Latched control decisions (FSM correctness — MANDATORY)

**Any decoded control value that is consumed outside the cycle that produced
it MUST be latched into a register. An FSM may never re-derive a decision from
a live bus or datapath signal in a later state.**

The recurring silicon bug this rule kills (three independent generations of it
have shipped past block DV): a combinational expression is only VALID during
one narrow window — e.g. `assign cmd = {hi_nibble_q, live_bus_nibble}` is the
command byte exactly on the second command-nibble cycle — but a later FSM
state (address phase, data phase, drain) re-reads the same expression, whose
live component now carries UNRELATED data. The decode silently changes:
a write opcode re-evaluated during the address phase becomes whatever the
address nibble makes it.

Rules:
1. At the cycle a multi-cycle decision completes (opcode assembled, length
   field finished, mode decoded), latch it: `decision_q <= decision_w;`.
   Every later state reads `decision_q`, never the combinational form.
2. Grep your own RTL before finishing: any `assign X_w = {..._q, live_input}`
   consumed in more than one FSM state is a defect unless X is re-derived
   fresh each cycle BY DESIGN (and then say so in a comment).
3. Block-level DV is structurally weak against this class (the model often
   shares the bug, and a TB can mask it by accident of address choice), so do
   not rely on a green block sim — apply the rule at WRITE time.
