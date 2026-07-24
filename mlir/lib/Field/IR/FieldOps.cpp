#include "Field/IR/FieldOps.h"
#include "Field/IR/FieldDialect.h"
#include "Field/IR/FieldTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace ffjit::field;

#define GET_OP_CLASSES
#include "Field/IR/FieldOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Canonicalization
//
// Field values are opaque SSA values, but `from_int(arith.constant c)` is a
// recognizable literal with known value c mod p. That is enough to implement
// the useful algebraic identities: x+0, x-0, x*1, x*0, x-x, -(-x), and
// inv(inv(x)) (valid at 0 too, by the inv(0) = 0 convention), plus constant
// folding of add/sub/mul/neg over literals. Combined with DCE of the Pure
// ops, mul-by-zero folding erases entire dead term chains -- e.g. the
// `a * Z^4` term of a generic Weierstrass doubling formula when a = 0.
//===----------------------------------------------------------------------===//

namespace {

/// If `v` is from_int(arith.constant c) of element type `elem`, yield the
/// canonical residue c mod p.
static bool matchLiteral(Value v, APInt &residue) {
  auto fi = v.getDefiningOp<FromIntOp>();
  if (!fi)
    return false;
  APInt c;
  if (!matchPattern(fi.getInput(), m_ConstantInt(&c)))
    return false;
  auto elem = cast<ElementType>(v.getType());
  APInt p = elem.getModulusValue();
  unsigned w = std::max(c.getBitWidth(), p.getBitWidth());
  residue = c.zext(w).urem(p.zext(w));
  return true;
}

/// Build from_int(arith.constant value) : elem at `loc`.
static Value makeLiteral(PatternRewriter &rewriter, Location loc,
                         ElementType elem, const APInt &value) {
  auto intTy = rewriter.getIntegerType(elem.getStorageBitWidth());
  APInt v = value.zextOrTrunc(elem.getStorageBitWidth());
  Value cst =
      rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, v));
  return rewriter.create<FromIntOp>(loc, elem, cst);
}

static APInt modulusOf(Value v, unsigned width) {
  return cast<ElementType>(v.getType()).getModulusValue().zextOrTrunc(width);
}

struct AddCanon : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    APInt l, r;
    bool lConst = matchLiteral(op.getLhs(), l);
    bool rConst = matchLiteral(op.getRhs(), r);
    // x + 0 -> x ; 0 + x -> x
    if (rConst && r.isZero()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (lConst && l.isZero()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    // literal + literal -> literal
    if (lConst && rConst) {
      unsigned w = std::max(l.getBitWidth(), r.getBitWidth()) + 1;
      APInt p = modulusOf(op.getResult(), w);
      APInt sum = (l.zext(w) + r.zext(w)).urem(p);
      rewriter.replaceOp(op, makeLiteral(rewriter, op.getLoc(),
                                         cast<ElementType>(op.getType()), sum));
      return success();
    }
    return failure();
  }
};

struct SubCanon : OpRewritePattern<SubOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    auto elem = cast<ElementType>(op.getType());
    // x - x -> 0
    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOp(
          op, makeLiteral(rewriter, op.getLoc(), elem, APInt::getZero(1)));
      return success();
    }
    APInt l, r;
    bool lConst = matchLiteral(op.getLhs(), l);
    bool rConst = matchLiteral(op.getRhs(), r);
    // x - 0 -> x
    if (rConst && r.isZero()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // literal - literal -> literal
    if (lConst && rConst) {
      unsigned w = std::max(l.getBitWidth(), r.getBitWidth()) + 1;
      APInt p = modulusOf(op.getResult(), w);
      APInt diff = (l.zext(w) + p - r.zext(w)).urem(p);
      rewriter.replaceOp(op, makeLiteral(rewriter, op.getLoc(), elem, diff));
      return success();
    }
    return failure();
  }
};

struct MulCanon : OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    APInt l, r;
    bool lConst = matchLiteral(op.getLhs(), l);
    bool rConst = matchLiteral(op.getRhs(), r);
    // x * 1 -> x ; x * 0 -> 0 (and mirrored)
    if (rConst) {
      if (r.isOne()) {
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
      if (r.isZero()) {
        rewriter.replaceOp(op, op.getRhs());
        return success();
      }
    }
    if (lConst) {
      if (l.isOne()) {
        rewriter.replaceOp(op, op.getRhs());
        return success();
      }
      if (l.isZero()) {
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
    }
    // literal * literal -> literal
    if (lConst && rConst) {
      unsigned w = l.getBitWidth() + r.getBitWidth();
      APInt p = modulusOf(op.getResult(), w);
      APInt prod = (l.zext(w) * r.zext(w)).urem(p);
      rewriter.replaceOp(op,
                         makeLiteral(rewriter, op.getLoc(),
                                     cast<ElementType>(op.getType()), prod));
      return success();
    }
    return failure();
  }
};

struct NegCanon : OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    // -(-x) -> x
    if (auto inner = op.getOperand().getDefiningOp<NegOp>()) {
      rewriter.replaceOp(op, inner.getOperand());
      return success();
    }
    // -literal -> literal
    APInt c;
    if (matchLiteral(op.getOperand(), c)) {
      unsigned w = c.getBitWidth() + 1;
      APInt p = modulusOf(op.getResult(), w);
      APInt neg = (p - c.zext(w)).urem(p);
      rewriter.replaceOp(op, makeLiteral(rewriter, op.getLoc(),
                                         cast<ElementType>(op.getType()), neg));
      return success();
    }
    return failure();
  }
};

struct InvCanon : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter &rewriter) const override {
    // inv(inv(x)) -> x. Valid everywhere: inv is an involution on GF(p)*,
    // and the inv(0) = 0 convention extends it to all of GF(p).
    if (auto inner = op.getOperand().getDefiningOp<InvOp>()) {
      rewriter.replaceOp(op, inner.getOperand());
      return success();
    }
    return failure();
  }
};

} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddCanon>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SubCanon>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MulCanon>(context);
}

void NegOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<NegCanon>(context);
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<InvCanon>(context);
}
