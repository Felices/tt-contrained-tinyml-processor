# neural_core.sdc — authored timing constraints for tt_um_felixcheng_neural_core
#
# Replaces the flow's generic fallback SDC with a real timing contract so STA
# signs off against meaningful constraints rather than a clock-only default.
#
# Clock: 50 MHz (20 ns). V0.8 pipelines the MAC (multiply -> product register ->
# accumulate), so the per-stage path closes 50 MHz. I/O delay models the TT
# harness (inputs registered at the carrier board): a realistic 4 ns budget.

set clk_period 20.0

# --- clock ---
create_clock -name clk -period $clk_period [get_ports clk]
set_clock_uncertainty 0.25 [get_clocks clk]

# --- signal integrity ---
# With the single-accumulator datapath there is a lot of setup slack, so the
# resizer would otherwise downsize cells to save area and under-drive high-fanout
# nets (slow-corner max-transition warnings). Pin a design-wide slew ceiling so it
# keeps drive strength up regardless of timing slack -> fully clean signoff.
# 1.2 ns is below sky130's ~1.5 ns default. At the ~78 % 1x1 utilization the
# high-fanout reset net (to ~143 flops) naturally sits ~1.14 ns with little room
# left to buffer it tighter, so the ceiling tracks up with density (1.0 → 1.1 →
# 1.2); 1.2 clears it with no over-buffering (logic nets stay well under).
set_max_transition 1.2 [current_design]

# --- input / output timing budget (relative to clk) ---
set io_budget 4.0
set_input_delay  $io_budget -clock clk [get_ports {ui_in[*] uio_in[*]}]
set_output_delay $io_budget -clock clk [get_ports {uo_out[*] uio_out[*]}]

# --- things that must not be timed as synchronous paths ---
# rst_n is an asynchronous, active-low reset.
set_false_path -from [get_ports rst_n]
# ena is a static "powered" enable, constant during operation.
set_false_path -from [get_ports ena]
