/*
 * tt_um_felixcheng_neural_core — V0.15 programmable neural compute core
 *
 * A minimal, byte-stream-driven int8 MAC accelerator: a streaming multiply-
 * accumulate engine with a single int32 accumulator, a pipelined MAC, and a
 * SPILL opcode that reads the raw accumulator out for the host.
 *
 * Design summary (full version-by-version evolution + rationale in JOURNEY.md):
 *   - single int32 accumulator; pipelined saturating 8x8 MAC (explicit radix-4
 *     Booth unit `booth_mult8`, exhaustively verified in test/test_booth.py).
 *   - streaming MACV dot-product primitive, plus a toggle-able wide-input mode
 *     (WMACV) that time-shares the uio pins for up to 2x throughput.
 *   - SPILL reads the raw int32 accumulator out, so the host owns bias / requant
 *     / activation (output-only math is offloaded; no on-chip requant tail).
 *   - MISR self-test signature + serial debug readback for bring-up.
 *
 * Copyright (c) 2026 Felix Cheng
 * SPDX-License-Identifier: Apache-2.0
 *
 * ============================ Pin protocol ==============================
 *   ui_in[7:0]   Normal mode : instruction/operand byte (consumed when in_valid)
 *                Debug  mode : ui_in[3:0] selects a debug register to read back
 *   uio_in[0]  = in_valid   (host: a byte is present on ui_in this cycle)
 *   uio_in[1]  = dbg_en     (host: read debug register instead of executing)
 *   uo_out[7:0]  Normal mode : latest SPILL byte;  Debug mode : debug register
 *   uio_out[2] = out_valid  (chip: uo_out holds a fresh SPILL byte, 1-cyc pulse)
 *   uio_out[3] = done       (chip: HALT or illegal-instruction reached)
 *   uio_out[4] = busy       (chip: mid multi-byte instruction, expecting operands)
 *   uio_out[5] = error      (chip: sticky illegal-instruction flag)
 *
 * ============================== ISA (V0.15) =============================
 *   0x00 NOP
 *   0x01 LOAD_ACT   + 1 operand byte (signed int8 activation)
 *   0x02 LOAD_W     + 1 operand byte (signed int8 weight)
 *   0x03 MAC                          ACC += act * weight  (saturating int32)
 *   0x04 MACV       + N, then 2N bytes (act,weight) pairs: ACC += sum a_i*w_i
 *   0x05 CLRACC                       ACC = 0
 *   0x06 SPILL                        emit raw int32 ACC as 4 LE bytes; the
 *                                     host does bias/requant/activation in software
 *   0x07 HALT                         done = 1
 *   0x08 WMACV      + N: wide streaming MAC. For N host-clocked cycles the uio
 *                        pins carry the weight byte (activation on ui_in), one
 *                        MAC/cycle -- 2x the throughput of MACV. uio_oe flips to
 *                        input for the burst; the status/debug pins return after.
 *   else -> illegal_instruction, error, done
 */

`default_nettype none

module tt_um_felixcheng_neural_core (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  // ---------------------------- ISA opcodes ----------------------------
  localparam [7:0] OP_NOP      = 8'h00;
  localparam [7:0] OP_LOAD_ACT = 8'h01;
  localparam [7:0] OP_LOAD_W   = 8'h02;
  localparam [7:0] OP_MAC      = 8'h03;
  localparam [7:0] OP_MACV     = 8'h04;
  localparam [7:0] OP_CLRACC   = 8'h05;
  localparam [7:0] OP_SPILL    = 8'h06;  // emit raw int32 ACC as 4 bytes
  localparam [7:0] OP_HALT     = 8'h07;
  localparam [7:0] OP_WMACV    = 8'h08;  // wide streaming MAC: N cycles of ui_in x uio

  // ----------------------------- FSM states ----------------------------
  // 9 states -> 4 bits.
  localparam [3:0] S_FETCH     = 4'd0;
  localparam [3:0] S_LDACT     = 4'd1;
  localparam [3:0] S_LDW       = 4'd2;
  localparam [3:0] S_MACV_N    = 4'd3;
  localparam [3:0] S_MACV_A    = 4'd4;
  localparam [3:0] S_MACV_W    = 4'd5;
  localparam [3:0] S_HALT      = 4'd6;
  localparam [3:0] S_ERR       = 4'd7;
  localparam [3:0] S_WMACV_RUN = 4'd8;   // wide streaming-MAC burst

  // int32 saturation bounds
  localparam signed [32:0] I32_MAX = 33'sd2147483647;   //  2^31 - 1
  localparam signed [32:0] I32_MIN = -33'sd2147483648;  // -2^31

  // ------------------------------ control -------------------------------
  wire       in_valid = uio_in[0];
  wire       dbg_en   = uio_in[1];
  reg  [3:0] state;
  reg        wide;                             // wide-MACV burst in progress soon
  // spill sequencer state (declared here because `consume` stalls on it below)
  reg        spill_req;
  reg        spill_active;
  reg  [1:0] spill_cnt;
  wire       halted   = (state == S_HALT) || (state == S_ERR);
  // Input is stalled while a SPILL is draining (spill_req/spill_active): the
  // readout streams directly out of `acc`, so `acc` must hold still. BUSY (below)
  // reflects this, so the host knows to wait.
  wire       consume  = in_valid && !dbg_en && !halted && !spill_req && !spill_active;
  // The datapath advances on a consumed opcode/operand (narrow) or on every clock
  // of a wide-MACV burst (host-clock-paced; the uio handshake bits are weight data).
  wire       advance  = consume || (state == S_WMACV_RUN);
  wire [7:0] b        = ui_in;

  // ------------------------ architectural state -------------------------
  reg signed [31:0] acc;                       // single int32 accumulator
  reg signed [7:0]  act_reg;
  reg signed [7:0]  w_reg;
  reg        [7:0]  out_reg;
  reg        [7:0]  pc;                         // debug: opcodes accepted
  reg        [7:0]  cyc;                        // debug: cycles since exec start
  reg        [7:0]  cur_instr;                  // debug: current opcode
  reg        [7:0]  macv_cnt;
  reg signed [7:0]  macv_act;
  reg        [5:0]  status;
  reg               out_valid_r;
  reg        [15:0] sig;                       // MISR over SPILL output bytes
  reg               pend;                       // pipelined product pending accumulation
  reg signed [15:0] pprod;
  // spill sequencer emits the raw int32 accumulator as 4 bytes, straight from
  // `acc` (no shadow copy — input is stalled during the drain, so acc is stable);
  // its state regs (spill_req/spill_active/spill_cnt) are declared up top.

  // status: [0] illegal [1] overflow [2] reserved [3] output
  //         [4] started [5] finished

  // -------------------- pipelined MAC datapath (int32) ------------------
  // Stage 1 (multiply) -> pprod register. Stage 2 (accumulate) commits pprod
  // into acc, saturating to int32. A pending product commits before any read
  // sees the accumulator, so no forwarding is needed.
  // In a wide-MACV burst the operands arrive in parallel: activation on ui_in,
  // weight on uio_in (the wide mode borrows the uio pins as the weight byte).
  wire signed [7:0]  mul_a = (state == S_WMACV_RUN) ? $signed(ui_in)  :
                             (state == S_MACV_W)    ? macv_act        : act_reg;
  wire signed [7:0]  mul_b = (state == S_WMACV_RUN) ? $signed(uio_in) :
                             (state == S_MACV_W)    ? $signed(b)      : w_reg;
  wire signed [15:0] mprod;
  booth_mult8 u_mult (.a(mul_a), .b(mul_b), .p(mprod));

  wire signed [32:0] psum     = $signed({acc[31], acc})
                              + $signed({{17{pprod[15]}}, pprod});
  wire               psum_ovf = (psum > I32_MAX) || (psum < I32_MIN);
  wire signed [31:0] psum_sat = (psum > I32_MAX) ? 32'sh7FFF_FFFF :
                                (psum < I32_MIN) ? 32'sh8000_0000 :
                                                   psum[31:0];

  // the SPILL byte being emitted this cycle (little-endian) — straight from acc
  wire [7:0] spill_byte = acc[{spill_cnt, 3'b000} +: 8];

  // 16-bit MISR (Galois LFSR, poly x^16+x^15+x^13+x^4) over SPILL output bytes.
  function [15:0] misr_next;
    input [15:0] s;
    input [7:0]  d;
    begin
      misr_next = ({s[14:0], 1'b0} ^ (s[15] ? 16'hB400 : 16'h0000)) ^ {8'h00, d};
    end
  endfunction

  // ------------------------------ sequencer -----------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_FETCH;
      acc         <= 32'sd0;
      act_reg     <= 8'sd0;
      w_reg       <= 8'sd0;
      out_reg     <= 8'd0;
      pc          <= 8'd0;
      cyc         <= 8'd0;
      cur_instr   <= 8'd0;
      macv_cnt    <= 8'd0;
      macv_act    <= 8'sd0;
      wide        <= 1'b0;
      status      <= 6'd0;
      out_valid_r <= 1'b0;
      sig         <= 16'd0;
      pend        <= 1'b0;
      pprod       <= 16'sd0;
      spill_req   <= 1'b0;
      spill_active<= 1'b0;
      spill_cnt   <= 2'd0;
    end else begin
      out_valid_r <= 1'b0;
      spill_req   <= 1'b0;

      // ---- spill output (free-running: 4 bytes, LE, folded into the MISR) ----
      if (spill_req) begin
        spill_active <= 1'b1;                   // input stalls while active; acc holds still

        spill_cnt    <= 2'd0;
      end
      if (spill_active) begin
        out_reg     <= spill_byte;
        out_valid_r <= 1'b1;
        status[3]   <= 1'b1;                    // output_generated
        sig         <= misr_next(sig, spill_byte);
        spill_cnt   <= spill_cnt + 2'd1;
        if (spill_cnt == 2'd3) spill_active <= 1'b0;   // 4 LE bytes (int32)
      end

      // free-running cycle counter once execution has begun (debug)
      if (status[4] && !halted)
        cyc <= cyc + 8'd1;

      if (advance) begin
        status[4] <= 1'b1;

        // ---- pipeline stage 2: commit the pending product (if any) ----
        if (pend) begin
          acc <= psum_sat;                     // may be overridden by CLRACC below
          if (psum_ovf) status[1] <= 1'b1;
        end
        pend <= 1'b0;

        case (state)
          S_FETCH: begin
            cur_instr   <= b;
            pc          <= pc + 8'd1;
            case (b)
              OP_NOP:      ;
              OP_LOAD_ACT: state <= S_LDACT;
              OP_LOAD_W:   state <= S_LDW;
              OP_MAC: begin pprod <= mprod; pend <= 1'b1; end
              OP_MACV:  begin wide <= 1'b0; state <= S_MACV_N; end
              OP_WMACV: begin wide <= 1'b1; state <= S_MACV_N; end
              OP_CLRACC:    acc <= 32'sd0;
              OP_SPILL:     spill_req <= 1'b1;
              OP_HALT: begin state <= S_HALT; status[5] <= 1'b1; end
              default: begin state <= S_ERR; status[0] <= 1'b1; status[5] <= 1'b1; end
            endcase
          end

          S_LDACT:   begin act_reg  <= $signed(b);  state <= S_FETCH; end
          S_LDW:     begin w_reg    <= $signed(b);  state <= S_FETCH; end

          S_MACV_N: begin
            macv_cnt <= b;
            state    <= (b == 8'd0) ? S_FETCH : (wide ? S_WMACV_RUN : S_MACV_A);
          end
          S_MACV_A: begin macv_act <= $signed(b); state <= S_MACV_W; end
          S_MACV_W: begin
            pprod <= mprod;  pend <= 1'b1;
            macv_cnt <= macv_cnt - 8'd1;
            state    <= (macv_cnt == 8'd1) ? S_FETCH : S_MACV_A;
          end

          // wide streaming MAC: 1 product/clock, activation on ui_in || weight on uio
          S_WMACV_RUN: begin
            pprod <= mprod;  pend <= 1'b1;
            macv_cnt <= macv_cnt - 8'd1;
            state    <= (macv_cnt == 8'd1) ? S_FETCH : S_WMACV_RUN;
          end

          default: state <= S_ERR;
        endcase
      end
    end
  end

  // ------------------------- debug readback mux -------------------------
  reg [7:0] dbg_data;
  always @(*) begin
    // Full debug readback (restored in V0.13 — the 1x1 die has the headroom, and
    // full observability strengthens both bring-up and the differential test).
    case (ui_in[3:0])
      4'h0:    dbg_data = pc;                           // opcodes accepted
      4'h1:    dbg_data = cur_instr;                    // current opcode
      4'h2:    dbg_data = {5'b0, state};                // FSM state (hang diagnosis)
      4'h4:    dbg_data = acc[7:0];                     // int32 accumulator, LE
      4'h5:    dbg_data = acc[15:8];
      4'h6:    dbg_data = acc[23:16];
      4'h7:    dbg_data = acc[31:24];                   // accumulator high byte (int32)
      4'h8:    dbg_data = act_reg;                      // activation register
      4'h9:    dbg_data = w_reg;                        // weight register
      4'hA:    dbg_data = out_reg;                      // latest SPILL byte
      4'hB:    dbg_data = {2'b0, status};               // [5:0]=status flags
      4'hC:    dbg_data = cyc;                          // cycle counter
      4'hE:    dbg_data = sig[7:0];                     // MISR self-test signature
      4'hF:    dbg_data = sig[15:8];
      default: dbg_data = 8'h00;                        // 0x3, 0xD unused
    endcase
  end

  // ------------------------------ outputs -------------------------------
  assign uo_out     = dbg_en ? dbg_data : out_reg;
  assign uio_out[0] = 1'b0;
  assign uio_out[1] = 1'b0;
  assign uio_out[2] = out_valid_r;
  assign uio_out[3] = halted;
  assign uio_out[4] = !halted && ((state != S_FETCH) || spill_req || spill_active);
  assign uio_out[5] = status[0];
  assign uio_out[6] = 1'b0;
  assign uio_out[7] = 1'b0;
  // In a wide-MACV burst all uio pins are inputs (the weight byte); otherwise
  // uio[5:2] drive status out and uio[1:0] are the control inputs.
  assign uio_oe     = (state == S_WMACV_RUN) ? 8'b0000_0000 : 8'b0011_1100;

  wire _unused = &{ena, 1'b0};   // uio_in[7:2] are now live (wide-MACV weight byte)

endmodule

// ===========================================================================
// Radix-4 (modified Booth) signed 8x8 -> 16 multiplier.
//
// A naive array multiplier forms one partial product per bit of the multiplier
// `b` -> 8 rows to sum. Booth recoding reads `b` two bits at a time (with a
// 1-bit overlap), encoding it into 4 signed digits in {-2,-1,0,+1,+2}. Each
// digit selects a trivial partial product (0, +-A, +-2A) placed at a 2-bit-
// spaced position. That halves the rows (8 -> 4) and shrinks the adder tree,
// which is the bulk of a multiplier's area -- at equal (single-cycle) speed.
//
// Signed operands are handled by the signed-digit encoding itself: the top
// group's -2*b[7] term contributes the multiplier's sign, and each partial
// product is sign-extended to 16 bits, so no extra correction row is needed.
//
// Exhaustively verified against `a*b` for all 65536 signed input pairs in
// test/test_booth.py.
// ===========================================================================
module booth_mult8 (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] p
);
  // 16-bit sign-extended (digit * A) for one Booth group, from its bit triplet
  // (b[2i+1], b[2i], b[2i-1]).
  function signed [15:0] booth_pp;
    input        [2:0] trip;
    input signed [7:0] aa;
    begin
      case (trip)
        3'b001, 3'b010: booth_pp =  $signed({{8{aa[7]}}, aa});        //  +A
        3'b011:         booth_pp =  $signed({{7{aa[7]}}, aa, 1'b0});  //  +2A
        3'b100:         booth_pp = -$signed({{7{aa[7]}}, aa, 1'b0});  //  -2A
        3'b101, 3'b110: booth_pp = -$signed({{8{aa[7]}}, aa});        //  -A
        default:        booth_pp =  16'sd0;                            // 000, 111 -> 0
      endcase
    end
  endfunction

  // Overlapping triplets of b (implicit b[-1] = 0), one per radix-4 digit.
  wire signed [15:0] pp0 = booth_pp({b[1], b[0], 1'b0}, a);   // position 0
  wire signed [15:0] pp1 = booth_pp({b[3], b[2], b[1]}, a);   // position 2
  wire signed [15:0] pp2 = booth_pp({b[5], b[4], b[3]}, a);   // position 4
  wire signed [15:0] pp3 = booth_pp({b[7], b[6], b[5]}, a);   // position 6

  assign p = pp0 + (pp1 <<< 2) + (pp2 <<< 4) + (pp3 <<< 6);
endmodule
