# Hardening signoff — Neural Compute Core V0.15

Local RTL→GDSII signoff for `tt_um_felixcheng_neural_core`, reproduced from source with
LibreLane 2.4.2 (the version `tt-gds-action@ttsky26c` pins). The design **fits on a single 1×1 tile** —
it occupied 1×2 from V0.5 through V0.11; two V0.12 area cuts (a bare-minimum debug readback + removing
the 24-bit SPILL shadow register) dropped the cell area under the one-tile threshold, halving the
shuttle cost. V0.13 restored the full debug port using the tile's headroom, and **V0.14 added a toggle-able
wide-input mode** (`WMACV`) that time-shares the `uio` pins as a 16-bit input for 2× throughput — full
debug and wide input never run at once, so they share the pins with no permanent sacrifice. What stays
on-chip: the pipelined Booth MAC, a single int32 accumulator, streaming `MACV`, the wide `WMACV` burst,
shadow-free `SPILL`, the MISR self-test (now covering both datapaths), and the full debug port.
**1485 cells** at 78.7 % of a 1×1 tile, **+5.99 ns** setup slack at 50 MHz, and a **100 % clean signoff**
— zero violations *and* zero warnings, antenna included. **V0.15 widened the accumulator int24 → int32**:
int24 overflowed at ~512 products (forcing the compiler to tile almost every layer), int32 holds ~131K,
so tiling is now rare and `SPILL` reconstruction stays only as the backup for extremely large reductions.
That pushed util to 78.7 % and required raising placement density to 82 % and the slew ceiling to 1.2 ns —
**the aggressive edge of the 1×1 tile, and the practical limit of what fits.**

## Reproduce (LibreLane 2.4.2, ttsky26c — canonical)

```bash
docker run --rm -v "$PWD":/work/project -v ~/.tt-pdk-cache:/work/pdk \
  -w /work/project --entrypoint bash \
  ghcr.io/librelane/librelane:2.4.2 /work/project/harden_librelane.sh
```

`harden_librelane.sh` builds TinyTapeout's merged config and invokes LibreLane natively; `tt/` is a
copy of tt-support-tools. Both `tt/` and `runs/` are gitignored — the CI `gds.yaml` action is the
authoritative signoff for the actual shuttle submission. (`harden_v03.sh` is the same flow targeting
`runs/verify` so a re-run never clobbers a reference `runs/wokwi`; the V0.3 signoff below is the
`runs/verify` run.)

## Signoff metrics (LibreLane 2.4.2, sky130A, 1×1 tile, 50 MHz, V0.15)

| Metric | Result | Verdict |
|---|---|---|
| Magic DRC | **0** | ✅ clean |
| KLayout DRC (2nd independent deck) | **0** | ✅ clean |
| Magic ↔ KLayout XOR | **0** (streamout GDS identical) | ✅ clean |
| Routing DRC | **0** | ✅ clean |
| Netgen LVS | **0** device/net/pin diffs (match unique) | ✅ clean |
| Setup timing | **0 violations** @ 20 ns, worst-corner slack **+5.99 ns** | ✅ meets @ 50 MHz |
| Hold timing | **0 violations** | ✅ meets |
| Core utilization | **78.7 %** (1×1 tile, 82 % placement density) | ✅ fits — aggressive edge |
| Std-cell instances | 1485 (143 sequential / flip-flops) | — |
| Antenna violations | **0** | ✅ clean |
| Max-slew / max-fanout | **0 / 0** | ✅ clean |
| `design__violations` (aggregate) | **0** | ✅ clean |

> **Zero** violations *and* zero warnings across DRC, LVS, XOR, timing, antenna, slew, and fanout.
> The slew story is a recurring one: whenever the design has abundant setup slack, the resizer
> downsizes cells to save area and under-drives high-fanout nets (slow-corner max-transition
> warnings). Two fixes appear in this project's history — reapplying clock pressure (V0.8: 50 MHz
> packs tighter and cleaner than 40 MHz) and, when the clock is already fixed, an authored
> `set_max_transition` ceiling in the SDC that pins drive strength regardless of slack. As utilization
> climbed (71.8 % → 73.8 % → 78.7 % across V0.13–V0.15) the high-fanout reset net's slew tracked up, so
> the ceiling followed it — 1.0 → 1.1 → 1.2 ns — all comfortably under sky130's ~1.5 ns default.

## Functional verification

- **RTL cocotb suite: 36 / 36 PASS** (icarus + cocotb 2.x) — every opcode, signed arithmetic +
  int32 saturation (~133 K-MAC overflow → saturate), streaming `MACV` (incl. `N=0` and `N=255`), **wide `WMACV`** (directed, edge
  operands, wide-equals-narrow equivalence, and `uio_oe` direction), edge-operand sweeps through both
  multiply paths, clear + re-accumulate, saturating-accumulator overflow, PC / cycle-counter wrap,
  back-pressure, illegal opcodes, the (full) debug interface, `SPILL` (direct readout with the
  input-stall + a 3-chunk host-reconstruction of a long reduction), the MISR self-test (covering **both**
  the narrow and wide datapaths), and a 60-seed randomized **differential** test (full-state agreement).
- **Exhaustive Booth check: PASS** — the radix-4 Booth multiplier (`booth_mult8`) is verified against
  `a*b` for **all 65 536** signed 8×8 input pairs (`make -f Makefile.booth`), including every sign
  corner (−128×−128, −128×127, …). No sampling, no shared golden model.
- **Gate-level sim: 35 / 35 PASS** on the V0.15 powered post-layout netlist with the sky130 cell
  models — `make GATES=yes PDK_ROOT=~/.tt-pdk-cache` (the ~500-cycle overflow stress test is RTL-only,
  `skip=GL`).
- Fresh yosys elaboration of the current source: **0 inferred latches**, `check -assert` clean.
- **Authored timing constraints** in `src/neural_core.sdc` (clock + I/O delays + async-reset
  false path), wired via `PNR_SDC_FILE` / `SIGNOFF_SDC_FILE` — real STA, not the generic fallback.

## Range-limit behavior

- **Accumulator overflow saturates** to `INT32_MAX` / `INT32_MIN` (no wrap) and sets sticky
  `status[1]`. Pinned by `test_accumulator_overflow`. For reductions beyond the ~131 K-product int32
  budget (rare), the host tiles them via `SPILL` + `CLRACC` and sums the raw partials (exact) — pinned
  by `test_spill_reconstruct_long_reduction`.

## How it got here

The design evolved through several versions (see `JOURNEY.md` for the full arc): V0.2 minimal MAC
→ V0.3 features (accumulators, bias, ReLU) → V0.4 pipelined for 50 MHz → V0.5 narrowed to 2× int24
+ `SPILL` (fits 1×2) → V0.6 cut the output-only requant tail to the host → V0.7 removed the MAC
pipeline (single-cycle, 40 MHz) → V0.8 put the pipeline back and returned to 50 MHz → V0.9 cut
the second accumulator + `SEL_ACC` → V0.10 made the multiply an explicit radix-4 Booth unit → V0.11
trimmed two debug-only registers → V0.12 cut the debug port to a bare minimum and removed the SPILL
shadow register, dropping under the one-tile threshold and **retargeting from 1×2 to 1×1** → V0.13
restored the full debug port using the freed tile headroom (63.7 % → 71.8 %) → **V0.14** added a
toggle-able wide-input mode (`WMACV`) that time-shares the `uio` pins for 2× throughput (→ 73.8 %) →
**V0.15** widened the accumulator int24 → int32 (rare tiling now; `SPILL` kept as backup), pushing to
78.7 % util at 82 % placement density — the practical 1×1 ceiling.
V0.8's gate-level accounting showed the pipeline register actually *shrinks* the design (a clean
synthesis boundary between multiply and accumulate). V0.9 applied a byte-level cost model to the
*interface*: without a broadcast-MAC instruction, two accumulators can't beat the host re-streaming
`MACV`, so the register file was dead area — cut for −2020 µm². V0.10 found the synthesizer's `*` was
already near-Booth (hand-Booth won only ~220 µm²). V0.12's two cuts freed another ~2330 µm² and carried
the design across the tile boundary — area is continuous, but value is discrete (a tile). What's left
is the interesting hardware (pipelined Booth MAC, single accumulator, streaming MACV, shadow-free spill,
MISR self-test, full debug port, wide-input mode, int32 accumulator) at **78.7 % of a single 1×1 tile**, 50 MHz, fully clean. The CI
`gds.yaml` action remains authoritative for the actual shuttle submission.
