from pathlib import Path

import pytest

import ffjit.compiler as compiler
import ffjit.runtime as runtime
from ffjit.errors import CompileError
from ffjit.mlirgen import GeneratedModule


class _FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.result


class _FakeLibrary:
    def __init__(self, abi):
        self.ff_rt_abi_version = _FakeFunction(abi)
        self.ff_rt_inv = _FakeFunction()
        self.ff_rt_msm_schedule = _FakeFunction()
        self.ff_rt_fixed_base_schedule = _FakeFunction()


def test_runtime_abi_constant():
    assert runtime.RUNTIME_ABI_VERSION == 3


def test_runtime_discovery_accepts_environment_directory(tmp_path, monkeypatch):
    library = tmp_path / "libff_rt.so"
    library.touch()
    monkeypatch.setenv("FFJIT_RUNTIME", str(tmp_path))

    assert runtime.find_runtime() == str(library.resolve())


def test_runtime_exposes_path(tmp_path, monkeypatch):
    library = tmp_path / "libff_rt.so"
    library.touch()
    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda path: _FakeLibrary(runtime.RUNTIME_ABI_VERSION),
    )

    loaded = runtime.Runtime(str(library))

    assert loaded.path == str(library.resolve())
    assert loaded.abi_version == runtime.RUNTIME_ABI_VERSION


def test_runtime_rejects_abi_mismatch(monkeypatch):
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda path: _FakeLibrary(1))

    with pytest.raises(RuntimeError, match="requires 3.*provides 1"):
        runtime.Runtime("libff_rt_old.so")


def _install_fake_toolchain(monkeypatch, tmp_path, native_commands):
    monkeypatch.setattr(compiler, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(compiler, "_find_ffc", lambda: "ffc")
    monkeypatch.setattr(compiler, "_find_clang", lambda: "clang")
    monkeypatch.setattr(
        compiler,
        "CompiledKernel",
        lambda so_path, mod: Path(so_path),
    )

    def fake_run(cmd, what):
        output = Path(cmd[cmd.index("-o") + 1])
        if what == "MLIR-to-LLVM lowering":
            output.write_text("declare void @kernel()\n")
        else:
            native_commands.append(cmd)
            output.touch()

    monkeypatch.setattr(compiler, "_run", fake_run)


def test_runtime_free_module_does_not_discover_or_link_runtime(tmp_path, monkeypatch):
    native_commands = []
    _install_fake_toolchain(monkeypatch, tmp_path, native_commands)

    def unexpected_discovery():
        raise AssertionError("runtime discovery must remain opt-in")

    monkeypatch.setattr(compiler, "find_runtime", unexpected_discovery)
    mod = GeneratedModule("func.func @kernel() -> i64\n", "kernel", [], [64])

    compiler.compile_module(mod)

    assert mod.requires_runtime is False
    assert all("ff_rt" not in arg for arg in native_commands[0])


def test_runtime_required_module_links_with_rpath(tmp_path, monkeypatch):
    native_commands = []
    _install_fake_toolchain(monkeypatch, tmp_path / "cache", native_commands)
    (tmp_path / "cache").mkdir()
    library = tmp_path / "runtime" / "libff_rt.so"
    library.parent.mkdir()
    library.touch()
    checked = []
    monkeypatch.setattr(compiler, "find_runtime", lambda: str(library))
    monkeypatch.setattr(compiler, "Runtime", lambda path: checked.append(path))
    mod = GeneratedModule(
        "func.func @kernel() -> i64\n",
        "kernel",
        [],
        [64],
        requires_runtime=True,
    )

    compiler.compile_module(mod)

    assert checked == [str(library)]
    assert str(library) in native_commands[0]
    assert f"-Wl,-rpath,{library.parent.resolve()}" in native_commands[0]


def test_compile_reports_runtime_abi_mismatch(monkeypatch):
    monkeypatch.setattr(compiler, "find_runtime", lambda: "libff_rt_old.so")

    def reject(path):
        raise RuntimeError("ffjit runtime ABI mismatch: requires 3, provides 1")

    monkeypatch.setattr(compiler, "Runtime", reject)
    mod = GeneratedModule("", "kernel", [], [64], requires_runtime=True)

    with pytest.raises(CompileError, match="compatible libff_rt.*ABI mismatch"):
        compiler.compile_module(mod)


def test_linked_kernel_runtime_abi_is_validated():
    matching = _FakeLibrary(runtime.RUNTIME_ABI_VERSION)
    compiler._validate_linked_runtime(matching, "kernel.so")

    stale = _FakeLibrary(runtime.RUNTIME_ABI_VERSION - 1)
    with pytest.raises(CompileError, match="runtime-backed kernel ABI mismatch"):
        compiler._validate_linked_runtime(stale, "kernel.so")


def test_runtime_path_participates_in_cache_key(tmp_path, monkeypatch):
    native_commands = []
    cache = tmp_path / "cache"
    cache.mkdir()
    _install_fake_toolchain(monkeypatch, cache, native_commands)
    monkeypatch.setattr(compiler, "Runtime", lambda path: None)
    first = tmp_path / "first" / "libff_rt.so"
    second = tmp_path / "second" / "libff_rt.so"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"runtime")
    second.write_bytes(b"runtime")
    selected = [str(first), str(second)]
    monkeypatch.setattr(compiler, "find_runtime", lambda: selected.pop(0))
    mod = GeneratedModule(
        "func.func @kernel() -> i64\n",
        "kernel",
        [],
        [64],
        requires_runtime=True,
    )

    first_kernel = compiler.compile_module(mod)
    second_kernel = compiler.compile_module(mod)

    assert first_kernel != second_kernel


def test_runtime_inversion_option_reaches_ffc_and_links(tmp_path, monkeypatch):
    native_commands = []
    lowering_commands = []
    _install_fake_toolchain(monkeypatch, tmp_path, native_commands)
    fake_run = compiler._run

    def record_run(cmd, what):
        if what == "MLIR-to-LLVM lowering":
            lowering_commands.append(cmd)
        fake_run(cmd, what)

    monkeypatch.setattr(compiler, "_run", record_run)
    library = tmp_path / "libff_rt.so"
    library.touch()
    monkeypatch.setattr(compiler, "find_runtime", lambda: str(library))
    monkeypatch.setattr(compiler, "Runtime", lambda path: None)
    linked_library = _FakeLibrary(runtime.RUNTIME_ABI_VERSION)
    bound = type("_BoundKernel", (), {"_lib": linked_library})()
    monkeypatch.setattr(compiler, "CompiledKernel", lambda so_path, mod: bound)
    validated = []
    monkeypatch.setattr(
        compiler,
        "_validate_linked_runtime",
        lambda loaded, so_path: validated.append((loaded, so_path)),
    )
    mod = GeneratedModule("func.func @kernel() -> i64\n", "kernel", [], [64])

    compiler.compile_module(mod, inv="runtime")

    assert "--inv=runtime" in lowering_commands[0]
    assert str(library) in native_commands[0]
    assert validated and validated[0][0] is linked_library


def test_limb_specialization_reaches_ffc_and_cache_key(tmp_path, monkeypatch):
    native_commands = []
    lowering_commands = []
    _install_fake_toolchain(monkeypatch, tmp_path, native_commands)
    fake_run = compiler._run

    def record_run(cmd, what):
        if what == "MLIR-to-LLVM lowering":
            lowering_commands.append(cmd)
        fake_run(cmd, what)

    monkeypatch.setattr(compiler, "_run", record_run)
    mod = GeneratedModule("func.func @kernel() -> i64\n", "kernel", [], [64])

    generic = compiler.compile_module(mod, limb_specialization="generic")
    compact = compiler.compile_module(mod, limb_specialization="compact")

    assert generic != compact
    assert "--limb-specialization=generic" in lowering_commands[0]
    assert "--limb-specialization=compact" in lowering_commands[1]
