"""pytest wrapper around the math audit.

    pip install pytest && python -m pytest tests -q

Every case is an identity from `recherche.tex` re-derived numerically by an
independent route.  If one of these fails, the code and the notes disagree —
which is exactly what we want a test suite to tell us.
"""
import pytest

from msflow import checks

RESULTS = checks.run_all()


@pytest.mark.parametrize("res", RESULTS, ids=[r["check"] for r in RESULTS])
def test_identity(res):
    assert res["passed"], (
        f"{res['check']}: error {res['error']:.3e} > tolerance {res['tolerance']:.3e}"
        f"  ({res['detail']})"
    )
