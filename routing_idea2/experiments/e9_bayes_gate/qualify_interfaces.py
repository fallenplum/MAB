"""Execute the task's small, real mathematical and interface test suite."""
from __future__ import annotations

import io
import unittest

from common import CODE, RESULTS, Timer, sha256, write_json


def main():
    stream = io.StringIO()
    with Timer() as timer:
        suite = unittest.defaultTestLoader.discover(str(CODE / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = RESULTS / "private/logs/SOFTWARE_QUALIFICATION.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(log)
    log.write_text(stream.getvalue(), encoding="utf-8")
    write_json(RESULTS / "01_software_qualification.json", dict(
        status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), timing=timer.record,
        log_sha256=sha256(log),
        note="Second recorded execution of the same small synthetic suite; first pre-freeze execution also passed 20 tests",
        source_sha256={str(p.relative_to(CODE)): sha256(p) for p in sorted((CODE / "tests").glob("test_*.py"))},
    ))
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
