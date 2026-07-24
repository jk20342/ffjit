// RUN: ff-opt --canonicalize %s | FileCheck %s

// Algebraic identities over literals (from_int of arith.constant) and
// involutions. Together with DCE of Pure ops, mul-by-zero erases entire
// dead term chains.

!e = !field.elem<65537 : i128>

// CHECK-LABEL: func.func @identities
// CHECK-SAME: (%[[X:.*]]: i128)
func.func @identities(%x: i128) -> i128 {
  %c0 = arith.constant 0 : i128
  %c1 = arith.constant 1 : i128
  %zero = field.from_int %c0 : i128 -> !e
  %one  = field.from_int %c1 : i128 -> !e
  // CHECK: %[[XE:.*]] = field.from_int %[[X]]
  %xe = field.from_int %x : i128 -> !e
  // CHECK-NOT: field.mul
  // CHECK-NOT: field.add
  %t0 = field.mul %xe, %zero : !e
  %t1 = field.add %xe, %t0 : !e
  %t2 = field.mul %t1, %one : !e
  %t3 = field.add %zero, %t2 : !e
  // CHECK-NOT: field.neg
  // CHECK-NOT: field.inv
  %n1 = field.neg %t3 : !e
  %n2 = field.neg %n1 : !e
  %i1 = field.inv %n2 : !e
  %i2 = field.inv %i1 : !e
  // CHECK: %[[R:.*]] = field.to_int %[[XE]]
  %ret = field.to_int %i2 : !e -> i128
  // CHECK: return %[[R]]
  return %ret : i128
}

// CHECK-LABEL: func.func @constant_folding
func.func @constant_folding(%x: i128) -> i128 {
  %c5 = arith.constant 5 : i128
  %c7 = arith.constant 7 : i128
  %five = field.from_int %c5 : i128 -> !e
  %seven = field.from_int %c7 : i128 -> !e
  %xe = field.from_int %x : i128 -> !e
  // 5*7 = 35 and 5-7 = -2 = 65535 fold at compile time.
  // CHECK-DAG: arith.constant 35 : i64
  // CHECK-DAG: arith.constant 65535 : i64
  // CHECK-NOT: field.mul
  // CHECK-NOT: field.sub
  %m = field.mul %five, %seven : !e
  %s = field.sub %five, %seven : !e
  // CHECK: field.add
  // CHECK: field.add
  %r0 = field.add %xe, %m : !e
  %r1 = field.add %r0, %s : !e
  %ret = field.to_int %r1 : !e -> i128
  return %ret : i128
}

// CHECK-LABEL: func.func @sub_self
func.func @sub_self(%x: i128) -> i128 {
  %xe = field.from_int %x : i128 -> !e
  // x - x -> 0; then x + 0 -> x, so only the boundary casts remain.
  // CHECK-NOT: field.sub
  // CHECK-NOT: field.add
  %z = field.sub %xe, %xe : !e
  %r = field.add %xe, %z : !e
  %ret = field.to_int %r : !e -> i128
  return %ret : i128
}
