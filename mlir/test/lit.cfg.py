import os
import lit.formats

config.name = "ffjit"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Locate build tools relative to the source tree.
_this = os.path.dirname(__file__)
_build_tools = os.path.abspath(os.path.join(_this, "..", "build", "tools"))
_llvm_bin = "/usr/lib/llvm-21/bin"

config.environment["PATH"] = os.pathsep.join(
    [
        os.path.join(_build_tools, "ff-opt"),
        os.path.join(_build_tools, "ffc"),
        _llvm_bin,
        os.environ.get("PATH", ""),
    ]
)

for tool in ["ff-opt", "ffc", "FileCheck"]:
    config.substitutions.append((tool, tool))
