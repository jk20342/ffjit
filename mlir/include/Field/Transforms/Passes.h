#ifndef FFJIT_FIELD_TRANSFORMS_PASSES_H
#define FFJIT_FIELD_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace ffjit {
namespace field {

#define GEN_PASS_DECL
#include "Field/Transforms/Passes.h.inc"

/// Register all Field transform passes with the global pass registry.
#define GEN_PASS_REGISTRATION
#include "Field/Transforms/Passes.h.inc"

} // namespace field
} // namespace ffjit

#endif // FFJIT_FIELD_TRANSFORMS_PASSES_H
