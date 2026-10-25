"""Build the vendored MIT solver plus a small bounded-search bridge."""

import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "vendor" / "2048-ai"
build = ROOT / "build"
build.mkdir(exist_ok=True)
compiler = shutil.which("clang++") or shutil.which("g++")
if compiler is None:
    raise SystemExit("A C++ compiler is required (Xcode Command Line Tools or g++)")
(build / "config.h").write_text(
    "#define HAVE_UNORDERED_MAP 1\n#define HAVE_SYS_TIME_H 1\n#define HAVE_GETTIMEOFDAY 1\n"
)
suffix = "dylib" if platform.system() == "Darwin" else "so"
command = [
    compiler,
    "-O3",
    "-std=c++11",
    "-fPIC",
    "-shared",
    "-I",
    str(build),
    "-I",
    str(source),
    str(source / "bridge.cpp"),
    "-o",
    str(build / f"lib2048.{suffix}"),
]
subprocess.run(command, check=True)
print(
    json.dumps(
        {
            "library": f"build/lib2048.{suffix}",
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        },
        indent=2,
    )
)
