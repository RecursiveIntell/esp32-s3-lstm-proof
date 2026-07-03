//
// SPDX-FileCopyrightText: 2026 RecursiveIntell
// SPDX-License-Identifier: MIT
//
// Aligned int8 x int8 SIMD dot kernel for already-unpacked int4 tiles.
// a2: input int8 pointer, 16-byte aligned
// a3: unpacked weight int8 pointer, 16-byte aligned
// a4: len_div16
// return: int32 accumulator in a2

    .text
    .align  4

    .type   dot_i4_i8_simd_esp32s3, @function
    .align  4
    .global dot_i4_i8_simd_esp32s3

dot_i4_i8_simd_esp32s3:
    entry   a1, 32

    ee.zero.accx
    beqz    a4, .Li4_done

    // Prime first pair.
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a3, 16
    addi            a4, a4, -1
    beqz            a4, .Li4_last

    loopgtz a4, .Li4_loop_end
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    ee.vld.128.ip   q1, a3, 16
.Li4_loop_end:

.Li4_last:
    ee.vmulas.s8.accx  q0, q1

.Li4_done:
    movi.n  a3, 0
    nop
    ee.srs.accx a2, a3, 0
    retw.n

    .size   dot_i4_i8_simd_esp32s3, . - dot_i4_i8_simd_esp32s3
