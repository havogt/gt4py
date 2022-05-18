# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


from pathlib import Path
from setuptools import Command, find_namespace_packages, setup


def local_pkg(pkg_name: str, subdir: str) -> str:
    """Returns a path to a local package."""
    return f"{pkg_name} @ file://{Path.cwd().parent / subdir}"


if __name__ == "__main__":
    setup(
        use_scm_version=False,
        install_requires=[local_pkg("gt4py-eve", "eve")]
        # packages=find_namespace_packages(include=["gt4py.*"]),
    )  # Disable setuptools_scm as a temporary workaround
