# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simulation evidence bound to exact input and result bytes."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET


def input_hashes(paths) -> dict[str, str]:
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in paths}


def capture(xml_path, inputs: dict[str, str]) -> dict:
    """Read actual executed tests; reject missing evidence or changed inputs."""
    if not inputs or input_hashes(inputs) != inputs:
        raise ValueError("Simulation inputs changed or no input manifest was supplied")
    path = Path(xml_path).resolve()
    data = path.read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed simulation XML: {exc}") from exc
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("Simulation results contain no executed tests")
    failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    skipped = sum(c.find("skipped") is not None and c.find("failure") is None
                  and c.find("error") is None for c in cases)
    return {"version": 1, "inputs": inputs, "results_xml": str(path),
            "results_sha256": hashlib.sha256(data).hexdigest(),
            "tests_total": len(cases), "tests_failed": failed, "tests_skipped": skipped,
            "tests_passed": len(cases) - failed - skipped,
            "passed": failed == 0 and skipped == 0}


def validate(receipt: dict) -> dict:
    if receipt.get("version") != 1:
        raise ValueError("Missing simulation evidence receipt")
    actual = capture(receipt["results_xml"], receipt["inputs"])
    if actual != receipt:
        raise ValueError("Simulation result bytes or recorded counts changed")
    return actual
