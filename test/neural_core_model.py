"""Golden software reference model for tt_um_felixcheng_neural_core (V0.9).

Consumes the exact same byte stream as the RTL and reproduces the architectural
state, so the cocotb testbench can do cycle-free differential checking.

A single int32 accumulator, pipelined saturating Booth MAC, wide-input mode (WMACV),
and SPILL (raw int32 readout as 4 LE bytes, folded into the MISR). The second
accumulator and SEL_ACC were removed (the register file couldn't beat host re-streaming).
The requantization tail (bias / shift / saturate / ReLU) is offloaded to the
host, so those ops are gone. Keep in lock-step with src/project.v.
"""


def s8(x):
    x &= 0xFF
    return x - 256 if x >= 128 else x


def s24(x):
    x &= 0xFFFFFF
    return x - (1 << 24) if x >= (1 << 23) else x


# ISA opcodes (must match project.v)
OP_NOP, OP_LOAD_ACT, OP_LOAD_W, OP_MAC = 0x00, 0x01, 0x02, 0x03
OP_MACV, OP_CLRACC, OP_SPILL, OP_HALT = 0x04, 0x05, 0x06, 0x07
OP_WMACV = 0x08   # wide streaming MAC — architecturally identical to MACV (the wide
                  # datapath only changes how the operand bytes are *delivered*)

INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def misr_next(s, d):
    shifted = ((s << 1) & 0xFFFF)
    if s & 0x8000:
        shifted ^= 0xB400
    return (shifted ^ (d & 0xFF)) & 0xFFFF

# status flag bit positions
F_ILLEGAL = 1 << 0
F_OVERFLOW = 1 << 1
F_RESERVED = 1 << 2
F_OUTPUT = 1 << 3
F_STARTED = 1 << 4
F_FINISHED = 1 << 5


class NeuralCore:
    def __init__(self):
        self.acc = 0             # single int32 accumulator
        self.act = 0
        self.w = 0
        self.out_reg = 0
        self.sig = 0             # MISR over SPILL output bytes
        self.pc = 0
        self.cur_instr = 0
        self.illegal = False
        self.overflow = False
        self.output_generated = False
        self.started = False
        self.finished = False
        self.halted = False

    def status(self):
        return ((F_ILLEGAL if self.illegal else 0)
                | (F_OVERFLOW if self.overflow else 0)
                | (F_OUTPUT if self.output_generated else 0)
                | (F_STARTED if self.started else 0)
                | (F_FINISHED if self.finished else 0))

    def _accumulate(self, a, w):
        full = self.acc + a * w
        if full > INT32_MAX:
            self.acc = INT32_MAX
            self.overflow = True
        elif full < INT32_MIN:
            self.acc = INT32_MIN
            self.overflow = True
        else:
            self.acc = full

    def run(self, stream):
        it = iter(stream)
        for byte in it:
            if self.halted:
                break
            op = byte & 0xFF
            self.started = True
            self.cur_instr = op
            self.pc = (self.pc + 1) & 0xFF

            if op == OP_NOP:
                pass
            elif op == OP_LOAD_ACT:
                self.act = s8(next(it))
            elif op == OP_LOAD_W:
                self.w = s8(next(it))
            elif op == OP_MAC:
                self._accumulate(self.act, self.w)
            elif op == OP_MACV or op == OP_WMACV:
                n = next(it) & 0xFF
                for _ in range(n):
                    a = s8(next(it))
                    ww = s8(next(it))
                    self._accumulate(a, ww)
            elif op == OP_CLRACC:
                self.acc = 0
            elif op == OP_SPILL:
                v = self.acc & 0xFFFFFFFF                  # int32 -> 4 little-endian bytes
                for i in range(4):
                    byte = (v >> (8 * i)) & 0xFF
                    self.sig = misr_next(self.sig, byte)
                    self.out_reg = byte                   # ends as the high byte
                self.output_generated = True
            elif op == OP_HALT:
                self.finished = True
                self.halted = True
            else:
                self.illegal = True
                self.finished = True
                self.halted = True
        return self

    def snapshot(self):
        return {
            "pc": self.pc,
            "cur_instr": self.cur_instr,
            "acc": self.acc & 0xFFFFFFFF,
            "act": self.act & 0xFF,
            "w": self.w & 0xFF,
            "out_reg": self.out_reg,
            "status": self.status(),
            "sig": self.sig & 0xFFFF,
        }
