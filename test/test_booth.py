# SPDX-FileCopyrightText: © 2026 Felix Cheng
# SPDX-License-Identifier: Apache-2.0
#
# Exhaustive equivalence test for the radix-4 Booth multiplier (booth_mult8):
# every one of the 65536 signed 8x8 input pairs is checked against `a*b`. This
# is the gold-standard verification for a small arithmetic unit — no sampling,
# no golden model that could share a bug, just the full input space vs. Python's
# reference multiply. Run with:  make -f Makefile.booth

import cocotb
from cocotb.triggers import Timer


def s16(x):
    x &= 0xFFFF
    return x - (1 << 16) if x >= (1 << 15) else x


@cocotb.test()
async def exhaustive_signed_8x8(dut):
    fails = 0
    for a in range(-128, 128):
        for b in range(-128, 128):
            dut.a.value = a & 0xFF
            dut.b.value = b & 0xFF
            await Timer(1, unit="ns")
            got = s16(int(dut.p.value))
            if got != a * b:
                fails += 1
                if fails <= 10:
                    dut._log.error(f"{a} * {b}: expected {a*b}, got {got}")
    assert fails == 0, f"{fails}/65536 Booth products wrong"
    dut._log.info("Booth 8x8: all 65536 signed pairs match a*b (incl. -128*-128, "
                  "-128*127, sign corners)")
