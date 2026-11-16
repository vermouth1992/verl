# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail if production code reintroduces the removed batch container."""

import os
from argparse import ArgumentParser
from pathlib import Path
SEARCH_WHITELIST = []

# Build the forbidden name without embedding it as one literal.
SEARCH_KEYWORDS = ["Data" "Proto"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        path_in_str = str(path.absolute())

        path_in_whitelist = False
        for sw in SEARCH_WHITELIST:
            sw = sw.replace("/", os.sep)
            if sw in path_in_str:
                print(f"[SKIP] File {path_in_str} is allowlisted.")
                path_in_whitelist = True
                break

        if path_in_whitelist:
            continue

        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

        found = next((keyword for keyword in SEARCH_KEYWORDS if keyword in file_content), None)
        print(
            f"[CHECK] File {path_in_str} legacy batch-container check: "
            f"{'success' if found is None else 'failed'}."
        )
        assert found is None, f"file {path_in_str} contains the removed batch container; use TensorDict directly."
