//===- ConvertFieldToArith.cpp --------------------------------------------===//
//
// Lowers the `field` dialect to `arith` on wide integers. The default path
// keeps values in the Montgomery domain and reduces products with REDC.
//
// References (see doc/THEORY.md):
//   - P. Montgomery, "Modular Multiplication Without Trial Division",
//     Math. Comp. 44 (1985).
//   - Handbook of Applied Cryptography, Ch. 14 (Alg. 14.32, 14.36).
//   - The Newton/Hensel iteration for -p^{-1} mod 2^W.
//
//===----------------------------------------------------------------------===//

#include "Field/IR/FieldDialect.h"
#include "Field/IR/FieldOps.h"
#include "Field/IR/FieldTypes.h"
#include "Field/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace ffjit::field;

namespace ffjit {
namespace field {
#define GEN_PASS_DEF_CONVERTFIELDTOARITH
#include "Field/Transforms/Passes.h.inc"
} // namespace field
} // namespace ffjit

//===----------------------------------------------------------------------===//
// Compile-time Montgomery parameters for a fixed prime p.
//===----------------------------------------------------------------------===//

namespace {

/// All constants derived from the modulus at compile time, plus helpers that
/// emit the corresponding arith IR. Values in the "storage" width W are field
/// representatives in [0, p); intermediate products use a wide type.
struct FieldLowering {
  unsigned W;    // storage bit width = 64 * nlimbs, R = 2^W > p
  unsigned Wbig; // generic 2W + 64, or compact 2W + 1
  APInt p;       // modulus, width W
  APInt pInv;    // (-p^{-1}) mod 2^W, width W
  APInt R2;      // 2^(2W) mod p, width W (value < p)
  bool montgomery;

  Type ti;   // IntegerType(W)
  Type tbig; // IntegerType(Wbig)

  FieldLowering(ElementType elem, bool mont, bool requestCompact,
                MLIRContext *ctx) {
    montgomery = mont;
    W = elem.getStorageBitWidth();
    p = elem.getModulusValue().zextOrTrunc(W);

    // T + m*p is less than 2*R*p, hence less than 2^(2W+1). Restrict this
    // compact representation to benchmark candidates with the Montgomery
    // preconditions checked from the type itself.
    bool supportedWidth = W == 256 || W == 384 || W == 448;
    bool validModulus = !p.isZero() && p[0] && p.getActiveBits() < W;
    bool compact =
        montgomery && requestCompact && supportedWidth && validModulus;
    Wbig = compact ? 2 * W + 1 : 2 * W + 64;
    ti = IntegerType::get(ctx, W);
    tbig = IntegerType::get(ctx, Wbig);

    // -- p^{-1} mod 2^W via Newton/Hensel lifting (doubles correct bits each
    //    step; starts exact to 1 bit because p is odd). All arithmetic is
    //    width-W wrapping, i.e. modulo 2^W.
    APInt x(W, 1);
    APInt two(W, 2);
    for (int i = 0; i < 12; ++i)
      x = x * (two - p * x);
    pInv = APInt(W, 0) - x; // -inv mod 2^W

    // -- R^2 mod p = 2^(2W) mod p.
    APInt rsq = APInt(Wbig, 1).shl(2 * W);
    APInt pB = p.zext(Wbig);
    R2 = rsq.urem(pB).trunc(W);
  }

  // --- constant materialization ---
  Value cst(OpBuilder &b, Location loc, const APInt &v, Type t) const {
    APInt vv = v.zextOrTrunc(cast<IntegerType>(t).getWidth());
    return b.create<arith::ConstantOp>(loc, t, IntegerAttr::get(t, vv));
  }
  Value cstTi(OpBuilder &b, Location loc, const APInt &v) const {
    return cst(b, loc, v, ti);
  }
  Value cstTbig(OpBuilder &b, Location loc, const APInt &v) const {
    return cst(b, loc, v, tbig);
  }

  Value zext(OpBuilder &b, Location loc, Value v, Type t) const {
    return b.create<arith::ExtUIOp>(loc, t, v);
  }
  Value trunc(OpBuilder &b, Location loc, Value v, Type t) const {
    return b.create<arith::TruncIOp>(loc, t, v);
  }

  /// Given `wide` known to be in [0, 2p), conditionally subtract p and
  /// truncate to the storage type (result in [0, p)).
  Value condSubAndTrunc(OpBuilder &b, Location loc, Value wide) const {
    Value pB = cstTbig(b, loc, p);
    Value ge =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, wide, pB);
    Value sub = b.create<arith::SubIOp>(loc, wide, pB);
    Value sel = b.create<arith::SelectOp>(loc, ge, sub, wide);
    return trunc(b, loc, sel, ti);
  }

  /// Montgomery reduction: given T in [0, R*p) as a `tbig` value, return
  /// T * R^{-1} mod p in [0, p) as a `ti` value.
  Value redc(OpBuilder &b, Location loc, Value T) const {
    // m = ((T mod R) * pInv) mod R    -- low W bits, done in width W (wraps)
    Value Tlow = trunc(b, loc, T, ti);
    Value pInvC = cstTi(b, loc, pInv);
    Value m = b.create<arith::MulIOp>(loc, Tlow, pInvC); // mod 2^W
    // t = (T + m*p) / R
    Value mBig = zext(b, loc, m, tbig);
    Value pB = cstTbig(b, loc, p);
    Value mp = b.create<arith::MulIOp>(loc, mBig, pB);
    Value sum = b.create<arith::AddIOp>(loc, T, mp);
    Value shamt = cstTbig(b, loc, APInt(Wbig, W));
    Value t = b.create<arith::ShRUIOp>(loc, sum, shamt); // in [0, 2p)
    return condSubAndTrunc(b, loc, t);
  }

  /// Multiply two storage-width representatives.
  Value mul(OpBuilder &b, Location loc, Value a, Value bb) const {
    if (!montgomery) {
      Value aB = zext(b, loc, a, tbig);
      Value bB = zext(b, loc, bb, tbig);
      Value prod = b.create<arith::MulIOp>(loc, aB, bB);
      Value pB = cstTbig(b, loc, p);
      Value r = b.create<arith::RemUIOp>(loc, prod, pB);
      return trunc(b, loc, r, ti);
    }
    Value aB = zext(b, loc, a, tbig);
    Value bB = zext(b, loc, bb, tbig);
    Value prod = b.create<arith::MulIOp>(loc, aB, bB);
    return redc(b, loc, prod);
  }

  Value square(OpBuilder &b, Location loc, Value a) const {
    return mul(b, loc, a, a);
  }

  /// (a + b) mod p, both in [0, p).
  Value add(OpBuilder &b, Location loc, Value a, Value bb) const {
    Value aB = zext(b, loc, a, tbig);
    Value bB = zext(b, loc, bb, tbig);
    Value s = b.create<arith::AddIOp>(loc, aB, bB); // < 2p
    return condSubAndTrunc(b, loc, s);
  }

  /// (a - b) mod p, both in [0, p).
  Value sub(OpBuilder &b, Location loc, Value a, Value bb) const {
    Value aB = zext(b, loc, a, tbig);
    Value bB = zext(b, loc, bb, tbig);
    Value pB = cstTbig(b, loc, p);
    Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, aB, bB);
    Value aPlusP = b.create<arith::AddIOp>(loc, aB, pB);
    Value base = b.create<arith::SelectOp>(loc, ge, aB, aPlusP);
    Value diff = b.create<arith::SubIOp>(loc, base, bB); // < p
    return trunc(b, loc, diff, ti);
  }

  /// (-a) mod p.
  Value neg(OpBuilder &b, Location loc, Value a) const {
    Value aB = zext(b, loc, a, tbig);
    Value pB = cstTbig(b, loc, p);
    Value zero = cstTbig(b, loc, APInt(Wbig, 0));
    Value isz =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, aB, zero);
    Value pm = b.create<arith::SubIOp>(loc, pB, aB);
    Value sel = b.create<arith::SelectOp>(loc, isz, zero, pm);
    return trunc(b, loc, sel, ti);
  }

  /// The Montgomery representative of 1, i.e. R mod p.
  /// to_mont(1) = REDC(1 * R2) = REDC(R2) = R mod p.
  Value montOne(OpBuilder &b, Location loc) const {
    Value one = cstTi(b, loc, APInt(W, 1));
    return toMont(b, loc, one);
  }

  /// Move a canonical representative (in [0,p)) into the working domain.
  Value toMont(OpBuilder &b, Location loc, Value a) const {
    if (!montgomery)
      return a;
    Value r2 = cstTi(b, loc, R2);
    return mul(b, loc, a, r2); // REDC(a * R2) = a*R mod p
  }

  /// Move a working-domain value back to a canonical representative.
  Value fromMont(OpBuilder &b, Location loc, Value a) const {
    if (!montgomery)
      return a;
    Value aB = zext(b, loc, a, tbig); // T = a * 1
    return redc(b, loc, aB);
  }

  /// Raise a value to a non-negative compile-time exponent. Rather than
  /// unroll the ladder, emit a compact left-to-right square-and-multiply
  /// `scf.for`. The multiply is predicated so the body is branch-free.
  Value pow(OpBuilder &b, Location loc, Value a, const APInt &exponent) const {
    APInt e = exponent.zextOrTrunc(std::max(W, exponent.getBitWidth()));
    unsigned nbits = std::max(1u, e.getActiveBits());
    Value one = montgomery ? montOne(b, loc) : cstTi(b, loc, APInt(W, 1));
    Value eC = cstTi(b, loc, e.zextOrTrunc(W));

    Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = b.create<arith::ConstantIndexOp>(loc, nbits);
    Value step = b.create<arith::ConstantIndexOp>(loc, 1);
    Value hiIdx = b.create<arith::ConstantIndexOp>(loc, nbits - 1);

    auto loop = b.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{one},
        [&](OpBuilder &bb, Location l, Value iv, ValueRange it) {
          Value res = it[0];
          Value sq = square(bb, l, res);
          // bit index counts down from the most significant bit of e.
          Value idx = bb.create<arith::SubIOp>(l, hiIdx, iv);
          Value shamt = bb.create<arith::IndexCastOp>(l, ti, idx);
          Value sh = bb.create<arith::ShRUIOp>(l, eC, shamt);
          Value bit = bb.create<arith::TruncIOp>(l, bb.getI1Type(), sh);
          Value prod = mul(bb, l, sq, a);
          Value next = bb.create<arith::SelectOp>(l, bit, prod, sq);
          bb.create<scf::YieldOp>(l, next);
        });
    return loop.getResult(0);
  }

  /// Multiplicative inverse via Fermat's little theorem: a^(p-2).
  Value inv(OpBuilder &b, Location loc, Value a) const {
    return pow(b, loc, a, p - APInt(W, 2));
  }

  /// Call the pointer-based C runtime on a canonical representative. The
  /// temporary wide integers have the runtime's little-endian limb layout on
  /// supported little-endian targets.
  Value runtimeInv(OpBuilder &b, Location loc, Value a) const {
    Value canonical = fromMont(b, loc, a);
    Value modulus = cstTi(b, loc, p);
    Value one = b.create<arith::ConstantIntOp>(loc, 1, 64);
    Type ptrTy = LLVM::LLVMPointerType::get(b.getContext());
    Value outPtr = b.create<LLVM::AllocaOp>(loc, ptrTy, ti, one, 8);
    Value inPtr = b.create<LLVM::AllocaOp>(loc, ptrTy, ti, one, 8);
    Value modPtr = b.create<LLVM::AllocaOp>(loc, ptrTy, ti, one, 8);
    b.create<LLVM::StoreOp>(loc, canonical, inPtr, 8);
    b.create<LLVM::StoreOp>(loc, modulus, modPtr, 8);
    Value limbs = b.create<arith::ConstantIntOp>(loc, W / 64, 64);
    b.create<LLVM::CallOp>(loc, TypeRange{}, "ff_rt_inv",
                           ValueRange{outPtr, inPtr, modPtr, limbs});
    Value result = b.create<LLVM::LoadOp>(loc, ti, outPtr, 8);
    return toMont(b, loc, result);
  }

  /// from_int: reduce an arbitrary-width unsigned integer into the domain.
  Value fromInt(OpBuilder &b, Location loc, Value input) const {
    Value inBig = zext(b, loc, input, tbig);
    Value pB = cstTbig(b, loc, p);
    Value red = b.create<arith::RemUIOp>(loc, inBig, pB);
    Value canon = trunc(b, loc, red, ti); // in [0, p)
    return toMont(b, loc, canon);
  }

  /// to_int: canonical representative, resized to the requested integer type.
  Value toInt(OpBuilder &b, Location loc, Value a, Type outTy) const {
    Value canon = fromMont(b, loc, a); // in [0, p), width W
    unsigned outW = cast<IntegerType>(outTy).getWidth();
    if (outW == W)
      return canon;
    if (outW > W)
      return b.create<arith::ExtUIOp>(loc, outTy, canon);
    return b.create<arith::TruncIOp>(loc, outTy, canon);
  }
};

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//

/// Build a FieldLowering for the element type of a field op result/operand.
static FieldLowering makeLowering(ElementType elem, bool mont, bool compact) {
  return FieldLowering(elem, mont, compact, elem.getContext());
}

struct FieldTypeConverter : public TypeConverter {
  FieldTypeConverter(MLIRContext *ctx) {
    addConversion([](Type t) { return t; });
    addConversion([ctx](ElementType e) -> Type {
      return IntegerType::get(ctx, e.getStorageBitWidth());
    });
  }
};

template <typename OpT> struct FieldOpPattern : OpConversionPattern<OpT> {
  FieldOpPattern(const TypeConverter &tc, MLIRContext *ctx, bool mont,
                 bool runtime, bool compact)
      : OpConversionPattern<OpT>(tc, ctx), montgomery(mont),
        runtimeInv(runtime), compactIntermediates(compact) {}
  bool montgomery;
  bool runtimeInv;
  bool compactIntermediates;
};

struct AddLowering : FieldOpPattern<AddOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(
        op, fl.add(rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct SubLowering : FieldOpPattern<SubOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(
        op, fl.sub(rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct MulLowering : FieldOpPattern<MulOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(
        op, fl.mul(rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct NegLowering : FieldOpPattern<NegOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(op, fl.neg(rewriter, op.getLoc(), adaptor.getOperand()));
    return success();
  }
};

struct InvLowering : FieldOpPattern<InvOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(InvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    if (runtimeInv && fl.W / 64 > 8)
      return op.emitOpError(
          "runtime inversion supports at most 8 64-bit limbs");
    Value result =
        runtimeInv ? fl.runtimeInv(rewriter, op.getLoc(), adaptor.getOperand())
                   : fl.inv(rewriter, op.getLoc(), adaptor.getOperand());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct PowLowering : FieldOpPattern<PowOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    APInt exponent(64, static_cast<uint64_t>(op.getExponent()));
    rewriter.replaceOp(
        op, fl.pow(rewriter, op.getLoc(), adaptor.getBase(), exponent));
    return success();
  }
};

struct FromIntLowering : FieldOpPattern<FromIntOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(FromIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getResult().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(op,
                       fl.fromInt(rewriter, op.getLoc(), adaptor.getInput()));
    return success();
  }
};

struct ToIntLowering : FieldOpPattern<ToIntOp> {
  using FieldOpPattern::FieldOpPattern;
  LogicalResult
  matchAndRewrite(ToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getInput().getType());
    auto fl = makeLowering(elem, montgomery, compactIntermediates);
    rewriter.replaceOp(op, fl.toInt(rewriter, op.getLoc(), adaptor.getInput(),
                                    op.getResult().getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

struct ConvertFieldToArithPass
    : public ffjit::field::impl::ConvertFieldToArithBase<
          ConvertFieldToArithPass> {
  using ffjit::field::impl::ConvertFieldToArithBase<
      ConvertFieldToArithPass>::ConvertFieldToArithBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();
    if (invMethod != "fermat" && invMethod != "runtime") {
      module.emitError("unknown inversion method '")
          << invMethod << "'; expected 'fermat' or 'runtime'";
      signalPassFailure();
      return;
    }
    if (limbSpecialization != "generic" && limbSpecialization != "auto" &&
        limbSpecialization != "compact") {
      module.emitError("unknown limb specialization '")
          << limbSpecialization
          << "'; expected 'generic', 'auto', or "
             "'compact'";
      signalPassFailure();
      return;
    }
    bool useRuntimeInv = invMethod == "runtime";
    // No compact variant is enabled automatically until benchmarks establish
    // a repeatable win. "compact" remains an explicit opt-in experiment.
    bool useCompactIntermediates = limbSpecialization == "compact";
    if (useRuntimeInv) {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(module.getBody());
      Type ptrTy = LLVM::LLVMPointerType::get(ctx);
      Type i64Ty = builder.getI64Type();
      Type voidTy = LLVM::LLVMVoidType::get(ctx);
      auto fnTy = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty}, false);
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "ff_rt_inv", fnTy);
    }

    FieldTypeConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<FieldDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();

    // func ops are legal once their signatures contain no field types.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp, func::CallOp>(
        [&](Operation *op) { return converter.isLegal(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<AddLowering, SubLowering, MulLowering, NegLowering,
                 InvLowering, PowLowering, FromIntLowering, ToIntLowering>(
        converter, ctx, useMontgomery, useRuntimeInv, useCompactIntermediates);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
