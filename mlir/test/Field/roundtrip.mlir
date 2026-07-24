// RUN: ff-opt %s | ff-opt | FileCheck %s

// CHECK-LABEL: func.func @small
func.func @small(%a: !field.elem<65537 : i32>, %b: !field.elem<65537 : i32>) -> !field.elem<65537 : i32> {
  // CHECK: field.add
  %0 = field.add %a, %b : !field.elem<65537 : i32>
  // CHECK: field.mul
  %1 = field.mul %0, %a : !field.elem<65537 : i32>
  // CHECK: field.sub
  %2 = field.sub %1, %b : !field.elem<65537 : i32>
  // CHECK: field.neg
  %3 = field.neg %2 : !field.elem<65537 : i32>
  // CHECK: field.inv
  %4 = field.inv %3 : !field.elem<65537 : i32>
  return %4 : !field.elem<65537 : i32>
}

// CHECK-LABEL: func.func @bn254
func.func @bn254(%x: i256) -> i256 {
  // CHECK: field.from_int
  %e = field.from_int %x : i256 -> !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
  %s = field.mul %e, %e : !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
  // CHECK: field.to_int
  %r = field.to_int %s : !field.elem<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256> -> i256
  return %r : i256
}
