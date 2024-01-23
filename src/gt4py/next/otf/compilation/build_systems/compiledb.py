# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
from typing import Optional, TypeVar

from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import build_data, cache, compiler
from gt4py.next.otf.compilation.build_systems import cmake, cmake_lists


SrcL = TypeVar("SrcL", bound=languages.NanobindSrcL)


@dataclasses.dataclass
class CompiledbFactory(
    compiler.BuildSystemProjectGenerator[
        SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python
    ]
):
    """
    Create a CompiledbProject from a ``CompilableSource`` stage object with given CMake settings.

    Use CMake to generate a compiledb with the required sequence of build commands.
    Generate a compiledb only if there isn't one for the given combination of cmake configuration and library dependencies.
    """

    cmake_build_type: cmake.BuildType = cmake.BuildType.RELEASE
    cmake_extra_flags: list[str] = dataclasses.field(default_factory=list)
    renew_compiledb: bool = False

    def __call__(
        self,
        source: stages.CompilableSource[
            SrcL,
            languages.LanguageWithHeaderFilesSettings,
            languages.Python,
        ],
        cache_strategy: cache.Strategy,
    ) -> CompiledbProject:
        if not source.binding_source:
            raise NotImplementedError(
                "Compiledb build system project requires separate bindings code file."
            )
        name = source.program_source.entry_point.name
        header_name = f"{name}.{source.program_source.language_settings.header_extension}"
        bindings_name = f"{name}_bindings.{source.program_source.language_settings.file_extension}"

        cc_prototype_program_source = _cc_prototype_program_source(
            deps=source.library_deps,
            build_type=self.cmake_build_type,
            cmake_flags=self.cmake_extra_flags or [],
            language=source.program_source.language,
            language_settings=source.program_source.language_settings,
        )

        if self.renew_compiledb or not (
            compiledb_template := _cc_find_compiledb(cc_prototype_program_source, cache_strategy)
        ):
            compiledb_template = _cc_create_compiledb(
                cc_prototype_program_source,
                build_type=self.cmake_build_type,
                cmake_flags=self.cmake_extra_flags or [],
                cache_strategy=cache_strategy,
            )

        return CompiledbProject(
            root_path=cache.get_cache_folder(source, cache_strategy),
            program_name=name,
            source_files={
                header_name: source.program_source.source_code,
                bindings_name: source.binding_source.source_code,
            },
            bindings_file_name=bindings_name,
            compile_commands_cache=compiledb_template,
        )


@dataclasses.dataclass()
class CompiledbProject(
    stages.BuildSystemProject[SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python]
):
    """
    Compiledb build system for gt4py programs.

    Rely on a pre-configured compiledb to run the right build steps in the right order.
    The advantage is that overall build time grows linearly in number of distinct configurations
    and not in number of GT4Py programs. In cases where many programs can reuse the same configuration,
    this can save multiple seconds per program over rerunning CMake configuration each time.

    Works independently of what is used to generate the compiledb.
    """

    root_path: pathlib.Path
    source_files: dict[str, str]
    program_name: str
    compile_commands_cache: pathlib.Path
    bindings_file_name: str

    def build(self) -> None:
        self._write_files()
        current_data = build_data.read_data(self.root_path)
        if current_data is None or current_data.status < build_data.BuildStatus.CONFIGURED:
            self._run_config()
            current_data = build_data.read_data(self.root_path)  # update after config
        if (
            current_data is not None
            and build_data.BuildStatus.CONFIGURED
            <= current_data.status
            < build_data.BuildStatus.COMPILED
        ):
            self._run_build()

    def _write_files(self) -> None:
        def ignore_not_libraries(folder: str, children: list[str]) -> list[str]:
            pattern = r"((lib.*\.a)|(.*\.lib))"
            libraries = [child for child in children if re.match(pattern, child)]
            folders = [child for child in children if (pathlib.Path(folder) / child).is_dir()]
            ignored = list(set(children) - set(libraries) - set(folders))
            return ignored

        shutil.copytree(
            self.compile_commands_cache.parent,
            self.root_path,
            ignore=ignore_not_libraries,
            dirs_exist_ok=True,
        )

        for name, content in self.source_files.items():
            (self.root_path / name).write_text(content, encoding="utf-8")

        build_data.write_data(
            data=build_data.BuildData(
                status=build_data.BuildStatus.INITIALIZED,
                module=pathlib.Path(""),
                entry_point_name=self.program_name,
            ),
            path=self.root_path,
        )

    def _run_config(self) -> None:
        compile_db = json.loads(self.compile_commands_cache.read_text())

        (self.root_path / "build").mkdir(exist_ok=True)
        (self.root_path / "build" / "bin").mkdir(exist_ok=True)

        for entry in compile_db:
            for key, value in entry.items():
                entry[key] = (
                    value.replace("$NAME", self.program_name)
                    .replace("$BINDINGS_FILE", self.bindings_file_name)
                    .replace("$SRC_PATH", str(self.root_path))
                )

        (self.root_path / "compile_commands.json").write_text(json.dumps(compile_db))

        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.CONFIGURED,
                module=pathlib.Path(compile_db[-1]["directory"]) / compile_db[-1]["output"],
                entry_point_name=self.program_name,
            ),
            self.root_path,
        )

    def _run_build(self) -> None:
        logfile = self.root_path / "log_build.txt"
        compile_db = json.loads((self.root_path / "compile_commands.json").read_text())
        assert compile_db
        try:
            with logfile.open(mode="w") as log_file_pointer:
                for entry in compile_db:
                    log_file_pointer.write(entry["command"] + "\n")
                    subprocess.check_call(
                        entry["command"],
                        cwd=entry["directory"],
                        shell=True,
                        stdout=log_file_pointer,
                        stderr=log_file_pointer,
                    )
        except subprocess.CalledProcessError as e:
            with logfile.open(mode="r") as log_file_pointer:
                print(log_file_pointer.read())
            raise e

        build_data.update_status(new_status=build_data.BuildStatus.COMPILED, path=self.root_path)


def _cc_prototype_program_name(
    deps: tuple[interface.LibraryDependency, ...], build_type: str, flags: list[str]
) -> str:
    base_name = "compile_commands_cache"
    deps_str = "_".join(f"{dep.name}_{dep.version}" for dep in deps)
    flags_str = "_".join(re.sub(r"\W+", "", f) for f in flags)
    return "_".join([base_name, deps_str, build_type, flags_str]).replace(".", "_")


def _cc_prototype_program_source(
    deps: tuple[interface.LibraryDependency, ...],
    build_type: cmake.BuildType,
    cmake_flags: list[str],
    language: type[SrcL],
    language_settings: languages.LanguageWithHeaderFilesSettings,
) -> stages.ProgramSource:
    name = _cc_prototype_program_name(deps, build_type.value, cmake_flags)
    return stages.ProgramSource(
        entry_point=interface.Function(name=name, parameters=()),
        source_code="",
        library_deps=deps,
        language=language,
        language_settings=language_settings,
    )


def _cc_find_compiledb(
    prototype_program_source: stages.ProgramSource, cache_strategy: cache.Strategy
) -> Optional[pathlib.Path]:
    cache_path = cache.get_cache_folder(
        stages.CompilableSource(prototype_program_source, None), cache_strategy
    )
    compile_db_path = cache_path / "compile_commands.json"
    if compile_db_path.exists():
        return compile_db_path
    return None


def _cc_create_compiledb(
    prototype_program_source: stages.ProgramSource,
    build_type: cmake.BuildType,
    cmake_flags: list[str],
    cache_strategy: cache.Strategy,
) -> pathlib.Path:
    name = prototype_program_source.entry_point.name
    cache_path = cache.get_cache_folder(
        stages.CompilableSource(prototype_program_source, None), cache_strategy
    )

    header_ext = prototype_program_source.language_settings.header_extension
    src_ext = prototype_program_source.language_settings.file_extension
    prog_src_name = f"{name}.{header_ext}"
    binding_src_name = f"{name}.{src_ext}"
    cmake_languages = [cmake_lists.Language(name="CXX")]
    if prototype_program_source.language is languages.Cuda:
        cmake_languages = [*cmake_languages, cmake_lists.Language(name="CUDA")]

    prototype_project = cmake.CMakeProject(
        generator_name="Ninja",
        build_type=build_type,
        extra_cmake_flags=cmake_flags,
        root_path=cache_path,
        source_files={
            **{name: "" for name in [binding_src_name, prog_src_name]},
            "CMakeLists.txt": cmake_lists.generate_cmakelists_source(
                name,
                prototype_program_source.library_deps,
                [binding_src_name, prog_src_name],
                cmake_languages,
            ),
        },
        program_name=name,
    )

    prototype_project.build()

    log_file = cache_path / "log_compiledb.txt"

    with log_file.open("w") as log_file_pointer:
        commands_json_str = subprocess.check_output(
            ["ninja", "-t", "compdb"],
            cwd=cache_path / "build",
            stderr=log_file_pointer,
        ).decode("utf-8")
        commands = json.loads(commands_json_str)

    compile_db = [
        cmd for cmd in commands if name in pathlib.Path(cmd["file"]).stem and cmd["command"]
    ]

    assert compile_db

    for entry in compile_db:
        entry["directory"] = entry["directory"].replace(str(cache_path), "$SRC_PATH")
        entry["command"] = (
            entry["command"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(str(cache_path), "$SRC_PATH")
            .replace(binding_src_name, "$BINDINGS_FILE")
            .replace(name, "$NAME")
            .replace("-I$SRC_PATH/build/_deps", f"-I{cache_path}/build/_deps")
        )
        entry["file"] = (
            entry["file"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(str(cache_path), "$SRC_PATH")
            .replace(binding_src_name, "$BINDINGS_FILE")
        )
        entry["output"] = (
            entry["output"]
            .replace(f"CMakeFiles/{name}.dir", ".")
            .replace(binding_src_name, "$BINDINGS_FILE")
            .replace(name, "$NAME")
        )

    compile_db_path = cache_path / "compile_commands.json"
    compile_db_path.write_text(json.dumps(compile_db))
    return compile_db_path
