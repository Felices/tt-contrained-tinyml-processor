# SPDX-FileCopyrightText: © 2026 Felix Cheng
# SPDX-License-Identifier: Apache-2.0
#
# Layered testbench for tt_um_felixcheng_neural_core (V0.9).
# The requant tail is offloaded to the host, so this suite covers the on-chip
# hardware: the pipelined saturating MAC, streaming MACV, the single int32
# accumulator, SPILL (raw readout + MISR), the debug port, and a 60-seed
# randomized differential test against the golden model.

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

from neural_core_model import (
    NeuralCore,
    OP_NOP, OP_LOAD_ACT, OP_LOAD_W, OP_MAC, OP_MACV, OP_CLRACC, OP_SPILL, OP_HALT,
    OP_WMACV,
)

# Gate-level sim (make GATES=yes) skips the one ~500-cycle stress test below.
GL = os.environ.get("GATES") == "yes"

# debug register addresses
# Full debug map (V0.13).
DBG_PC, DBG_CURI, DBG_STATE = 0x0, 0x1, 0x2
DBG_ACC0, DBG_ACC1, DBG_ACC2, DBG_ACC3 = 0x4, 0x5, 0x6, 0x7   # 0x7 = sign-ext byte
DBG_ACT, DBG_W, DBG_OUT, DBG_STATUS, DBG_CYC = 0x8, 0x9, 0xA, 0xB, 0xC
DBG_SIG0, DBG_SIG1 = 0xE, 0xF

_clk_handle = None

# Settle time before sampling combinational outputs (gate-level cell delay >> RTL);
# still a tiny fraction of the 10 us clock period.
SETTLE_NS = 100

PROD_MAX = 127 * 127  # 16129


# --------------------------------------------------------------------------
# low-level driver helpers
# --------------------------------------------------------------------------
async def start_clock(dut):
    global _clk_handle
    if _clk_handle is not None:
        _clk_handle.cancel()
    _clk_handle = cocotb.start_soon(Clock(dut.clk, 10, unit="us").start())


async def reset(dut):
    await start_clock(dut)
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)


async def send(dut, byte):
    dut.ui_in.value = byte & 0xFF
    dut.uio_in.value = 0b01  # in_valid=1, dbg_en=0
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0


async def run_program(dut, prog):
    for byte in prog:
        await send(dut, byte)
    dut.uio_in.value = 0


async def settle_pipe(dut, n=4):
    """Flush the MAC pipeline (the accumulate is one cycle behind the multiply);
    NOPs commit a pending product. Advances PC by n."""
    await run_program(dut, [OP_NOP] * n)


async def hold(dut, byte, n):
    """Hold one single-byte opcode with in_valid for n clocks (n executions)."""
    dut.ui_in.value = byte & 0xFF
    dut.uio_in.value = 0b01
    await ClockCycles(dut.clk, n)
    dut.uio_in.value = 0


async def idle(dut, n=1):
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, n)


async def read_dbg(dut, addr):
    dut.uio_in.value = 0b10  # dbg_en=1
    dut.ui_in.value = addr & 0xF
    await Timer(SETTLE_NS, unit="ns")
    return int(dut.uo_out.value)


def _decode_flags(v):
    return {"out_valid": (v >> 2) & 1, "done": (v >> 3) & 1,
            "busy": (v >> 4) & 1, "error": (v >> 5) & 1}


async def get_flags(dut):
    await Timer(SETTLE_NS, unit="ns")
    return _decode_flags(int(dut.uio_out.value))


async def _read_acc(dut):
    acc = 0
    for i, addr in enumerate((DBG_ACC0, DBG_ACC1, DBG_ACC2, DBG_ACC3)):
        acc |= (await read_dbg(dut, addr)) << (8 * i)
    return acc


async def snapshot(dut):
    snap = {
        "pc": await read_dbg(dut, DBG_PC),
        "cur_instr": await read_dbg(dut, DBG_CURI),
        "acc": await _read_acc(dut),
        "act": await read_dbg(dut, DBG_ACT),
        "w": await read_dbg(dut, DBG_W),
        "out_reg": await read_dbg(dut, DBG_OUT),
        "status": await read_dbg(dut, DBG_STATUS),
        "sig": (await read_dbg(dut, DBG_SIG1)) << 8 | (await read_dbg(dut, DBG_SIG0)),
    }
    dut.uio_in.value = 0
    return snap


async def check_against_model(dut, prog, label=""):
    """Run prog on DUT + model and assert full architectural-state agreement.
    Appends NOPs to flush the MAC pipeline so committed DUT state matches the
    latency-free model."""
    await reset(dut)
    drained = list(prog) + [OP_NOP] * 4
    await run_program(dut, drained)
    got = await snapshot(dut)
    exp = NeuralCore().run(drained).snapshot()
    for key in exp:
        assert got[key] == exp[key], (
            f"[{label}] mismatch on {key}: DUT={got[key]} MODEL={exp[key]}\n"
            f"  prog={prog}\n  DUT={got}\n  MODEL={exp}"
        )


async def wide_burst(dut, pairs):
    """Drive a WMACV: opcode + N (narrow), then N host-clocked cycles with the
    activation on ui_in and the weight on the uio pins (the wide weight byte)."""
    await send(dut, OP_WMACV)
    await send(dut, len(pairs))
    for a, w in pairs:
        dut.ui_in.value = a & 0xFF
        dut.uio_in.value = w & 0xFF            # all 8 uio bits carry the weight
        await ClockCycles(dut.clk, 1)
    dut.ui_in.value = 0
    dut.uio_in.value = 0


def wide_stream(pairs):
    """The equivalent flat byte stream for the golden model."""
    return [OP_WMACV, len(pairs)] + [b & 0xFF for p in pairs for b in p]


async def spill_bytes(dut, timeout=8):
    """Issue SPILL and collect the 4 raw int32 bytes over OUT_VALID pulses."""
    await send(dut, OP_SPILL)
    got = []
    for _ in range(timeout):
        await ClockCycles(dut.clk, 1)
        await Timer(SETTLE_NS, unit="ns")
        if ((int(dut.uio_out.value) >> 2) & 1) == 1:
            got.append(int(dut.uo_out.value))
            if len(got) == 4:
                break
    return got


# --------------------------------------------------------------------------
# L1 — reset
# --------------------------------------------------------------------------
@cocotb.test()
async def test_reset(dut):
    await reset(dut)
    snap = await snapshot(dut)
    assert snap["pc"] == 0 and snap["acc"] == 0 and snap["out_reg"] == 0 and snap["status"] == 0
    assert (await read_dbg(dut, DBG_STATE)) == 0
    f = await get_flags(dut)
    assert f["done"] == 0 and f["error"] == 0 and f["busy"] == 0


@cocotb.test()
async def test_reset_during_execution(dut):
    await reset(dut)
    await run_program(dut, [OP_LOAD_ACT, 7, OP_LOAD_W, 6, OP_MAC])
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 42
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    snap = await snapshot(dut)
    assert snap["acc"] == 0 and snap["pc"] == 0 and snap["status"] == 0


# --------------------------------------------------------------------------
# L2 — instruction decode
# --------------------------------------------------------------------------
@cocotb.test()
async def test_isa_nop(dut):
    await check_against_model(dut, [OP_NOP, OP_NOP], "NOP")


@cocotb.test()
async def test_isa_load_regs(dut):
    await check_against_model(dut, [OP_LOAD_ACT, 25, OP_LOAD_W, 0xF9], "LOAD")  # -7


@cocotb.test()
async def test_isa_mac(dut):
    await check_against_model(dut, [OP_LOAD_ACT, 4, OP_LOAD_W, 3, OP_MAC], "MAC")


@cocotb.test()
async def test_isa_clracc(dut):
    await check_against_model(dut, [OP_LOAD_ACT, 10, OP_LOAD_W, 10, OP_MAC, OP_CLRACC], "CLRACC")


@cocotb.test()
async def test_isa_halt(dut):
    await reset(dut)
    await run_program(dut, [OP_HALT])
    assert (await get_flags(dut))["done"] == 1
    assert (await read_dbg(dut, DBG_STATUS)) & (1 << 5)
    before = await read_dbg(dut, DBG_PC)
    await run_program(dut, [OP_LOAD_ACT, 5])           # ignored after HALT
    assert (await read_dbg(dut, DBG_PC)) == before


# --------------------------------------------------------------------------
# L3 — arithmetic
# --------------------------------------------------------------------------
@cocotb.test()
async def test_mac_signed(dut):
    for a, w in [(3, 4), (0xFD, 4), (0xFD, 0xFC), (127, 127), (0x80, 127), (0x80, 0x80)]:
        await check_against_model(dut, [OP_LOAD_ACT, a, OP_LOAD_W, w, OP_MAC], f"MAC {a}x{w}")


@cocotb.test()
async def test_mac_zero(dut):
    for a, w in [(0, 0), (0, 127), (0x80, 0)]:
        await check_against_model(dut, [OP_LOAD_ACT, a, OP_LOAD_W, w, OP_MAC], "zero")


# Edge operands driven through BOTH multiply paths (scalar MAC uses act_reg/w_reg;
# streaming MACV uses macv_act and the b-bus) so the Booth unit is exercised at the
# sign corners in-situ, on top of its standalone exhaustive test (test_booth.py).
EDGE_VALS = [0x80, 0x81, 0xFF, 0x00, 0x01, 0x7F, 0x40, 0xC0, 0x55, 0xAA]


@cocotb.test()
async def test_mac_edge_operands(dut):
    for a in EDGE_VALS:
        for w in EDGE_VALS:
            await check_against_model(
                dut, [OP_LOAD_ACT, a, OP_LOAD_W, w, OP_MAC], f"MAC edge {a:#04x}x{w:#04x}")


@cocotb.test()
async def test_macv_edge_operands(dut):
    for a in EDGE_VALS:
        for w in EDGE_VALS:
            await check_against_model(
                dut, [OP_MACV, 1, a, w], f"MACV edge {a:#04x}x{w:#04x}")


@cocotb.test()
async def test_accumulation(dut):
    prog = [OP_CLRACC,
            OP_LOAD_ACT, 2, OP_LOAD_W, 3, OP_MAC,
            OP_LOAD_ACT, 4, OP_LOAD_W, 5, OP_MAC,
            OP_LOAD_ACT, 6, OP_LOAD_W, 7, OP_MAC]      # 6+20+42 = 68
    await reset(dut)
    await run_program(dut, prog)
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 68
    await check_against_model(dut, prog, "accumulate")


# --------------------------------------------------------------------------
# L4 — streaming MACV / pipeline
# --------------------------------------------------------------------------
@cocotb.test()
async def test_macv_dotproduct(dut):
    prog = [OP_MACV, 4, 1, 5, 2, 6, 3, 7, 4, 8]        # [1,2,3,4].[5,6,7,8] = 70
    await reset(dut)
    await run_program(dut, prog)
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 70
    await check_against_model(dut, prog, "MACV dot")


@cocotb.test()
async def test_macv_negative_and_empty(dut):
    await check_against_model(dut, [OP_MACV, 2, 0xFF, 3, 0xFE, 4], "MACV neg")  # -1*3 + -2*4 = -11
    await check_against_model(dut, [OP_MACV, 0, OP_NOP], "MACV empty")


@cocotb.test()
async def test_macv_busy_flag(dut):
    await reset(dut)
    await send(dut, OP_MACV)
    await send(dut, 3)
    assert (await get_flags(dut))["busy"] == 1, "busy asserted mid-MACV"
    await run_program(dut, [1, 5, 2, 6, 3, 7])
    assert (await get_flags(dut))["busy"] == 0, "busy clears when vector complete"


@cocotb.test()
async def test_streaming_backtoback(dut):
    prog = [OP_MACV, 2, 2, 3, 4, 5,                    # 6+20 = 26
            OP_CLRACC,
            OP_MACV, 2, 1, 1, 2, 2]                    # 1+4 = 5
    await reset(dut)
    await run_program(dut, prog)
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 5, "second computation must not inherit"
    await check_against_model(dut, prog, "back-to-back")


@cocotb.test()
async def test_idle_cycles_no_phantom(dut):
    await reset(dut)
    await send(dut, OP_LOAD_ACT)
    await send(dut, 5)
    await send(dut, OP_LOAD_W)
    await send(dut, 5)
    await idle(dut, 5)                                 # bus idle: must not compute
    await send(dut, OP_MAC)
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 25, "idle cycles produced phantom MACs"


@cocotb.test()
async def test_macv_max_length(dut):
    prog = [OP_MACV, 255] + [1, 1] * 255               # dot = 255
    await reset(dut)
    await run_program(dut, prog)
    await settle_pipe(dut, 1)
    assert (await read_dbg(dut, DBG_ACC0)) == 255
    await check_against_model(dut, prog, "MACV N=255")


# --------------------------------------------------------------------------
# L4b — wide-input mode (WMACV): uio pins time-shared as the weight byte
# --------------------------------------------------------------------------
@cocotb.test()
async def test_wmacv(dut):
    pairs = [(2, 3), (4, 5), (1, 1), (7, 0xFE)]        # 6+20+1-14 = 13
    await reset(dut)
    await wide_burst(dut, pairs)
    await settle_pipe(dut, 2)                          # drain the last pipelined product
    exp = NeuralCore().run(wide_stream(pairs)).acc & 0xFFFFFFFF
    assert (await _read_acc(dut)) == exp, "WMACV accumulation"


@cocotb.test()
async def test_wmacv_matches_macv(dut):
    # Same operands via the wide path and the narrow path -> identical result.
    pairs = [(9, 0xF8), (127, 127), (0x80, 3), (5, 5)]
    await reset(dut)
    await wide_burst(dut, pairs)
    await settle_pipe(dut, 2)
    wide_acc = await _read_acc(dut)
    await reset(dut)
    narrow = [OP_MACV, len(pairs)] + [b & 0xFF for p in pairs for b in p]
    await run_program(dut, narrow)
    await settle_pipe(dut, 1)
    assert (await _read_acc(dut)) == wide_acc, "wide result must equal narrow"


@cocotb.test()
async def test_wmacv_uio_direction(dut):
    # uio_oe must flip to all-input during the burst and back to status after.
    await reset(dut)
    await send(dut, OP_WMACV)
    await send(dut, 2)                                 # now in the wide burst
    await Timer(SETTLE_NS, unit="ns")
    assert int(dut.uio_oe.value) == 0x00, "uio all-input during wide burst"
    dut.ui_in.value = 3;  dut.uio_in.value = 4
    await ClockCycles(dut.clk, 1)
    dut.ui_in.value = 1;  dut.uio_in.value = 1
    await ClockCycles(dut.clk, 1)                      # burst done -> back to narrow
    dut.ui_in.value = 0;  dut.uio_in.value = 0
    await Timer(SETTLE_NS, unit="ns")
    assert int(dut.uio_oe.value) == 0x3C, "uio status pins restored after burst"


@cocotb.test()
async def test_wmacv_edge_operands(dut):
    for a in EDGE_VALS:
        for w in EDGE_VALS:
            await reset(dut)
            await wide_burst(dut, [(a, w)])
            await settle_pipe(dut, 2)
            exp = NeuralCore().run(wide_stream([(a, w)])).acc & 0xFFFFFFFF
            assert (await _read_acc(dut)) == exp, f"WMACV edge {a:#04x}x{w:#04x}"


# --------------------------------------------------------------------------
# L5 — accumulator (clear + re-accumulate)
# --------------------------------------------------------------------------
@cocotb.test()
async def test_clracc_reaccumulate(dut):
    prog = [OP_LOAD_ACT, 3, OP_LOAD_W, 4, OP_MAC,       # acc = 12
            OP_CLRACC,                                   # acc = 0
            OP_LOAD_ACT, 5, OP_LOAD_W, 6, OP_MAC]        # acc = 30
    await reset(dut)
    await run_program(dut, prog)
    await settle_pipe(dut, 1)
    assert (await _read_acc(dut)) == 30, "acc cleared then re-accumulated"
    await check_against_model(dut, prog, "clracc reaccumulate")


# --------------------------------------------------------------------------
# L6 — illegal + range limits
# --------------------------------------------------------------------------
@cocotb.test()
async def test_illegal_opcode(dut):
    await reset(dut)
    await run_program(dut, [0xFF])
    f = await get_flags(dut)
    assert f["error"] == 1 and f["done"] == 1
    assert (await read_dbg(dut, DBG_STATUS)) & 1
    await check_against_model(dut, [0x55, 0x00], "illegal 0x55")


@cocotb.test(skip=GL)
async def test_accumulator_overflow(dut):
    # ~133K MACs of 127*127 to overflow int32; saturates + sets the sticky flag.
    n = ((1 << 31) - 1) // PROD_MAX + 2
    await reset(dut)
    await run_program(dut, [OP_LOAD_ACT, 127, OP_LOAD_W, 127])
    await hold(dut, OP_MAC, n)
    assert (await read_dbg(dut, DBG_STATUS)) & (1 << 1), "overflow flag sticky-set"
    exp = NeuralCore().run([OP_LOAD_ACT, 127, OP_LOAD_W, 127] + [OP_MAC] * n)
    assert await _read_acc(dut) == (exp.acc & 0xFFFFFFFF)
    assert exp.acc == (1 << 31) - 1, "saturates to INT32_MAX"


# --------------------------------------------------------------------------
# L7 — debug
# --------------------------------------------------------------------------
@cocotb.test()
async def test_debug_registers(dut):
    await reset(dut)
    await run_program(dut, [OP_LOAD_ACT, 9, OP_LOAD_W, 0xFB, OP_MAC])  # 9 * -5 = -45
    assert (await read_dbg(dut, DBG_ACT)) == 9
    assert (await read_dbg(dut, DBG_W)) == 0xFB
    assert (await read_dbg(dut, DBG_CURI)) == OP_MAC
    assert (await read_dbg(dut, DBG_PC)) == 3
    await settle_pipe(dut, 1)                          # commit the pipelined MAC
    assert (await _read_acc(dut)) == (-45 & 0xFFFFFFFF)


@cocotb.test()
async def test_pc_wraps(dut):
    await reset(dut)
    await hold(dut, OP_NOP, 260)
    assert (await read_dbg(dut, DBG_PC)) == 4, "PC wraps at 256"


@cocotb.test()
async def test_cycle_counter(dut):
    await reset(dut)
    c0 = await read_dbg(dut, DBG_CYC)
    await run_program(dut, [OP_NOP, OP_NOP, OP_NOP])
    assert (await read_dbg(dut, DBG_CYC)) > c0


@cocotb.test()
async def test_cycle_counter_wraps(dut):
    await reset(dut)
    await hold(dut, OP_NOP, 250)
    c1 = await read_dbg(dut, DBG_CYC)
    await hold(dut, OP_NOP, 20)
    c2 = await read_dbg(dut, DBG_CYC)
    assert ((c2 - c1) & 0xFF) == 20


@cocotb.test()
async def test_debug_priority_over_execute(dut):
    await reset(dut)
    dut.uio_in.value = 0b11                             # dbg_en=1 AND in_valid=1
    dut.ui_in.value = OP_LOAD_ACT
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    assert (await read_dbg(dut, DBG_PC)) == 0, "dbg_en blocks execution"
    assert (await read_dbg(dut, DBG_STATE)) == 0


@cocotb.test()
async def test_busy_during_scalar_load(dut):
    await reset(dut)
    await send(dut, OP_LOAD_ACT)
    assert (await get_flags(dut))["busy"] == 1
    await send(dut, 5)
    assert (await get_flags(dut))["busy"] == 0


@cocotb.test()
async def test_reset_mid_macv(dut):
    await reset(dut)
    await send(dut, OP_MACV)
    await send(dut, 4)
    await send(dut, 7)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    snap = await snapshot(dut)
    assert snap["acc"] == 0 and snap["pc"] == 0 and snap["status"] == 0
    assert (await read_dbg(dut, DBG_STATE)) == 0


# --------------------------------------------------------------------------
# L8 — SPILL (raw readout + host reconstruction)
# --------------------------------------------------------------------------
@cocotb.test()
async def test_spill(dut):
    prog = [OP_MACV, 2, 100, 100, 100, 100]            # dot = 20000
    await reset(dut)
    await run_program(dut, prog)
    exp_acc = NeuralCore().run(prog).acc & 0xFFFFFFFF
    got = await spill_bytes(dut)
    assert len(got) == 4, f"SPILL must emit 4 bytes, got {got}"
    recon = got[0] | (got[1] << 8) | (got[2] << 16) | (got[3] << 24)
    assert recon == exp_acc, f"spilled {recon:#010x} != acc {exp_acc:#010x}"


@cocotb.test()
async def test_spill_reconstruct_long_reduction(dut):
    # SPILL is the backup for reductions beyond the (now int32) budget: tile into
    # chunks, spill + CLRACC between them, host sums the raw int32 partials. Exact.
    chunks = [[(50, 40)] * 6, [(30, 30)] * 5, [(-20, 10)] * 4]   # 12000 + 4500 - 800
    await reset(dut)
    total = 0
    for ch in chunks:
        prog = [OP_MACV, len(ch)]
        for a, w in ch:
            prog += [a & 0xFF, w & 0xFF]
        await run_program(dut, prog)
        got = await spill_bytes(dut)
        raw = got[0] | (got[1] << 8) | (got[2] << 16) | (got[3] << 24)
        if raw & 0x80000000:
            raw -= (1 << 32)
        total += raw
        await run_program(dut, [OP_CLRACC])
    assert total == 12000 + 4500 - 800, f"reconstructed reduction wrong: {total}"


@cocotb.test()
async def test_misr_selftest(dut):
    # A fixed program + SPILL produces a deterministic MISR signature (over the
    # spilled bytes); matches the golden model exactly. Post-fab bring-up check.
    # Exercises BOTH datapaths: a narrow MACV and a WIDE WMACV, each spilled and
    # folded into the MISR, so a defect in either changes the signature.
    prog = ([OP_MACV, 3, 2, 3, 4, 5, 1, 1, OP_SPILL,   # narrow dot = 27, spill
             OP_CLRACC]
            + wide_stream([(10, 10), (1, 1)])          # wide dot = 101, spill
            + [OP_SPILL, OP_HALT])
    await reset(dut)
    # SPILL stalls input during its 3-cycle drain, so issue each SPILL via
    # spill_bytes() (which waits for the drain) rather than streaming it inline.
    await run_program(dut, [OP_MACV, 3, 2, 3, 4, 5, 1, 1])
    await spill_bytes(dut)                             # SPILL #1 -> acc = 27 (narrow path)
    await run_program(dut, [OP_CLRACC])
    await wide_burst(dut, [(10, 10), (1, 1)])          # exercise the WIDE datapath
    await settle_pipe(dut, 2)
    await spill_bytes(dut)                             # SPILL #2 -> acc = 101 (wide result)
    await run_program(dut, [OP_HALT])
    await settle_pipe(dut, 2)
    dut_sig = (await read_dbg(dut, DBG_SIG1)) << 8 | (await read_dbg(dut, DBG_SIG0))
    gold = NeuralCore().run(prog).sig
    assert dut_sig == gold, f"self-test signature: DUT={dut_sig:#06x} golden={gold:#06x}"
    assert dut_sig != 0


# --------------------------------------------------------------------------
# L9 — randomized differential
# --------------------------------------------------------------------------
def random_program(rng, n_instr):
    prog = []
    for _ in range(n_instr):
        op = rng.choice([OP_NOP, OP_LOAD_ACT, OP_LOAD_W, OP_MAC,
                         OP_MACV, OP_CLRACC])
        prog.append(op)
        if op in (OP_LOAD_ACT, OP_LOAD_W):
            prog.append(rng.randint(0, 255))
        elif op == OP_MACV:
            n = rng.randint(0, 4)
            prog.append(n)
            for _ in range(2 * n):
                prog.append(rng.randint(0, 255))
    return prog


@cocotb.test()
async def test_random_differential(dut):
    rng = random.Random(0xC0FFEE)
    for seed in range(60):
        prog = random_program(rng, rng.randint(3, 12))
        await check_against_model(dut, prog, f"random seed={seed}")
        assert (await read_dbg(dut, DBG_STATE)) == 0, f"seed {seed} left FSM mid-instruction"
