// RUN: ff-opt --convert-field-to-arith %s | FileCheck %s
// RUN: ff-opt --convert-field-to-arith='montgomery=false' %s | FileCheck %s --check-prefix=NAIVE

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

// from_int reduces modulo p (a remainder), then enters the Montgomery domain.
// CHECK-LABEL: func.func @entry
func.func @entry(%x: i64) -> i64 {
  // CHECK: arith.remui
  %e = field.from_int %x : i64 -> !field.elem<65537 : i64>
  %r = field.to_int %e : !field.elem<65537 : i64> -> i64
  return %r : i64
}
