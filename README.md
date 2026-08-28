![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Neural Compute Core (V0.15) — `tt_um_felixcheng_neural_core`

A tiny, byte-stream-programmable **int8 multiply-accumulate (MAC) core** on Tiny Tapeout
(sky130, 2026C shuttle). It is deliberately the *smallest* architecture that still gives a
compiler / MLIR backend a real hardware lowering target — an **ISA seed** targeted, after
fabrication, by a Python simulator, a compiler backend, and an MLIR experiment that all lower
to the same instruction set. It's a streaming int8 MAC engine with a single int32 accumulator and
a pipelined MAC at 50 MHz. `SPILL` reads the raw int32 accumulator out, and the **host** does the
requantization tail (bias / shift / saturate / activation) in software — a deliberate
hardware/software split that keeps the interesting silicon on-chip and the plain math off it.

> Full datasheet: [`docs/info.md`](docs/info.md).

## What it does

You stream one byte per clock into `ui_in` (gated by an `IN_VALID` handshake). The core
decodes a small instruction set and accumulates signed 8×8 products into a single **int32
accumulator**. The headline instruction is `MACV`, a streaming dot product — a quantized
`linear`/conv reduction tiles into a sequence of `MACV` ops. `SPILL`
reads the raw int32 accumulator out as 4 bytes; the **host** then applies bias, requant-shift,
saturate and activation. Because the output is the raw partial sum, reductions of *any* length
work — the host sums the partials across chunks, and tiles multiple output channels by re-issuing
`MACV` (cheaper over the byte-stream interface than an on-chip register file — see `JOURNEY.md`).

Everything needed to *drive and verify* the chip after fabrication is on-chip: a
non-intrusive serial **debug port** (read back the PC, FSM state, accumulator, status flags,
etc.) and a 16-bit **MISR self-test signature** so post-fab bring-up is a single
read-and-compare against a golden value.

## Instruction set (V0.15)

| Opcode | Name         | Operands             | Effect                                             |
|-------:|--------------|----------------------|----------------------------------------------------|
| `0x00` | `NOP`        | —                    | nothing                                            |
| `0x01` | `LOAD_ACT`   | 1 byte (int8)        | activation register = operand                      |
| `0x02` | `LOAD_W`     | 1 byte (int8)        | weight register = operand                          |
| `0x03` | `MAC`        | —                    | `ACC += act * weight` (saturating int32)      |
| `0x04` | `MACV`       | `N`, then `2N` bytes | `ACC += Σ actᵢ·wᵢ` — streaming dot product     |
| `0x05` | `CLRACC`     | —                    | `ACC = 0`                                     |
| `0x06` | `SPILL`      | —                    | emit raw int32 `ACC` as 4 LE bytes over `OUT_VALID`; MISR fold (host does bias/requant) |
| `0x07` | `HALT`       | —                    | stop; assert `DONE`                                |
| `0x08` | `WMACV`      | `N`                  | **wide** streaming MAC: `N` host-clocked cycles of act(`ui_in`)×weight(`uio`), 1 MAC/cycle — 2× `MACV` |
| other  | —            | —                    | illegal: sticky `ERROR`, assert `DONE`             |

## Pin protocol (24-pin TT harness)

| Pins | Direction | Meaning |
|------|-----------|---------|
| `ui_in[7:0]`  | in  | instruction/operand byte (normal); `ui_in[3:0]` = debug register address (debug mode) |
| `uo_out[7:0]` | out | latest `SPILL` byte (normal); selected debug register (debug mode) |
| `uio_in[0]`   | in  | `IN_VALID` — a byte is present on `ui_in` this cycle |
| `uio_in[1]`   | in  | `DBG_EN` — read a debug register instead of executing |
| `uio_out[2]`  | out | `OUT_VALID` — 1-cycle pulse when `uo_out` holds a fresh `SPILL` byte |
| `uio_out[3]`  | out | `DONE` — `HALT` or illegal instruction reached |
| `uio_out[4]`  | out | `BUSY` — mid multi-byte instruction, expecting operands |
| `uio_out[5]`  | out | `ERROR` — sticky illegal-instruction flag |

`uio_oe = 0011_1100` normally: `uio[5:2]` drive status out, `uio[1:0]` are control inputs. **During a
`WMACV` wide burst** `uio_oe = 0000_0000` — all 8 `uio` pins become inputs carrying the weight byte
(activation on `ui_in`), 1 MAC/clock; they revert to control/status after the burst. So wide input and
the debug/status pins **time-share** `uio` (you never need both at once). `BUSY` is asserted during a
wide burst and during a `SPILL`'s 3-cycle drain.
Clock: 50 MHz (pipelined MAC). Area: **1×1 tile**.

## How to test

A layered [cocotb](https://www.cocotb.org/) testbench checked against a golden Python
reference model (`test/neural_core_model.py` — itself the seed of the future Python
simulator) covers every opcode, signed arithmetic and saturation, streaming `MACV`,
`BUSY`/back-pressure behavior, illegal instructions, the debug interface, the wide-input mode,
and the MISR signature — plus a randomized **differential** test that runs many random
programs through both the RTL and the model and asserts full architectural-state agreement.

```sh
cd test
pip install -r requirements.txt
make
```

## Where this fits

This is the hardware half of a deliberately small **compiler/MLIR co-design stack**: freeze a
minimal ISA in silicon, then iterate the interesting software — the reference model, an
instruction lowering pass, an MLIR dialect that emits `MACV` — against a fixed, real target.
The companion gate-level `2x2 MAC` cell (`tiny-tapeout-mac2x2-wokwi`) is the single arithmetic
unit this core is built out of, expressed at the transistor/gate level.

## Status

- **V0.15** (pipelined int8 MAC engine, 50 MHz; single int32 accumulator; explicit radix-4 **Booth**
  multiplier; full debug port; **toggle-able wide-input mode**; shadow-free `SPILL`; host-side requant),
  hardened for the Tiny Tapeout **sky130 2026C** shuttle (`ttsky26c`) on a **single 1×1 tile** at
  78.7% util (82% placement density — the practical 1×1 ceiling), halving the 1×2 tile cost it
  occupied through V0.11. The int32 accumulator holds ~131K products, so the compiler rarely tiles;
  `SPILL` reconstruction stays as the backup for extremely large reductions.
- Verified: **36/36** RTL cocotb tests + gate-level sim + an **exhaustive** 65 536-pair Booth check,
  authored SDC, and a **fully clean** local RTL→GDS signoff (0 violations *and* 0 warnings) — see
  [`HARDENING.md`](HARDENING.md).

## License

Apache-2.0 · © 2026 Felix Cheng
