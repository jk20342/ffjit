"""Exception types for ffjit."""


class TraceError(TypeError):
    """Raised when a Python construct cannot be captured by the tracer."""


class CompileError(RuntimeError):
    """Raised when the MLIR/LLVM toolchain fails to compile a kernel."""
