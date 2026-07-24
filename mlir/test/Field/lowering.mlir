// RUN: ff-opt --convert-field-to-arith %s | FileCheck %s
// RUN: ff-opt --convert-field-to-arith='montgomery=false' %s | FileCheck %s --check-prefix=NAIVE
// RUN: ff-opt --convert-field-to-arith='inv=runtime' %s | FileCheck %s --check-prefix=RUNTIME
// RUN: ff-opt --pass-pipeline='builtin.module(convert-field-to-arith,cse)' %s | FileCheck %s --check-prefix=CSE
// RUN: ff-opt --convert-field-to-arith='limb-specialization=compact' %s | FileCheck %s --check-prefix=COMPACT
// RUN: ff-opt --convert-field-to-arith='limb-specialization=auto' %s | FileCheck %s --check-prefix=AUTO

// A word-sized prime: elem lowers to i64, wide intermediates to i192.
// CHECK-LABEL: func.func @addmul
// CHECK-SAME: (%{{.*}}: i64, %{{.*}}: i64) -> i64
func.func @addmul(%a: !field.elem<65537 : i64>, %b: !field.elem<65537 : i64>) -> !field.elem<65537 : i64> {
  // Montgomery multiply uses a wide product then REDC (shift by 64, no urem).
  // CHECK: arith.muli %{{.*}}, %{{.*}} : i192
  // CHECK: arith.shrui %{{.*}}, %{{.*}} : i192
  %0 = field.mul %a, %b : !field.elem<65537 : i64>
  // CHECK: arith.addi %{{.*}}, %{{.*}} : i192
  %1 = field.add %0, %a : !field.elem<65537 : i64>
  return %1 : !field.elem<65537 : i64>
}

// The naive path reduces multiplication with a remainder instead.
// NAIVE-LABEL: func.func @addmul
// NAIVE: arith.remui

// Inversion lowers to a fixed-trip-count scf.for (Fermat ladder), not unrolled.
// CHECK-LABEL: func.func @invert
func.func @invert(%a: !field.elem<65537 : i64>) -> !field.elem<65537 : i64> {
  // CHECK: scf.for
  // CHECK: scf.yield
  %0 = field.inv %a : !field.elem<65537 : i64>
  return %0 : !field.elem<65537 : i64>
}

// Runtime inversion converts from Montgomery form before the pointer call and
// converts the loaded canonical result back afterward.
// RUNTIME-LABEL: func.func @invert
// RUNTIME: arith.shrui
// RUNTIME: llvm.store
// RUNTIME: llvm.call @ff_rt_inv
// RUNTIME: llvm.load
// RUNTIME: arith.muli
// RUNTIME: arith.shrui

// A constant field power remains one compact loop regardless of exponent size.
// CHECK-LABEL: func.func @power
func.func @power(%a: !field.elem<65537 : i64>) -> !field.elem<65537 : i64> {
  // CHECK: scf.for
  // CHECK: scf.yield
  %0 = field.pow %a, 1234567 : !field.elem<65537 : i64>
  return %0 : !field.elem<65537 : i64>
}

// from_int reduces modulo p (a remainder), then enters the Montgomery domain.
// CHECK-LABEL: func.func @entry
func.func @entry(%x: i64) -> i64 {
  // CHECK: arith.remui
  %e = field.from_int %x : i64 -> !field.elem<65537 : i64>
  %r = field.to_int %e : !field.elem<65537 : i64> -> i64
  return %r : i64
}

// CSE-LABEL: func.func @duplicate
// CSE: %[[SUM:.*]] = arith.trunci
// CSE: return %[[SUM]], %[[SUM]]
func.func @duplicate(%a: !field.elem<65537 : i64>, %b: !field.elem<65537 : i64>) -> (!field.elem<65537 : i64>, !field.elem<65537 : i64>) {
  %0 = field.add %a, %b : !field.elem<65537 : i64>
  %1 = field.add %a, %b : !field.elem<65537 : i64>
  return %0, %1 : !field.elem<65537 : i64>, !field.elem<65537 : i64>
}

// Compact Montgomery intermediates retain exactly one carry bit.
// COMPACT-LABEL: func.func @compact4
// COMPACT-SAME: (%{{.*}}: i256, %{{.*}}: i256) -> i256
// COMPACT: arith.muli %{{.*}}, %{{.*}} : i513
// COMPACT: arith.shrui %{{.*}}, %{{.*}} : i513
// AUTO-LABEL: func.func @compact4
// AUTO: arith.muli %{{.*}}, %{{.*}} : i576
func.func @compact4(
    %a: !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>,
    %b: !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>)
    -> !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256> {
  %0 = field.mul %a, %b : !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
  return %0 : !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
}

// The 384-bit NIST P-384 modulus has a 448-bit (seven-limb) storage width.
// COMPACT-LABEL: func.func @compact7
// COMPACT-SAME: (%{{.*}}: i448, %{{.*}}: i448) -> i448
// COMPACT: arith.muli %{{.*}}, %{{.*}} : i897
// COMPACT: arith.shrui %{{.*}}, %{{.*}} : i897
// AUTO-LABEL: func.func @compact7
// AUTO: arith.muli %{{.*}}, %{{.*}} : i960
func.func @compact7(
    %a: !field.elem<39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815 : i384>,
    %b: !field.elem<39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815 : i384>)
    -> !field.elem<39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815 : i384> {
  %0 = field.mul %a, %b : !field.elem<39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815 : i384>
  return %0 : !field.elem<39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306815 : i384>
}
