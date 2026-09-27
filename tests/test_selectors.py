import pytest

from codeagentbench.verification.selectors import resolve_selectors


def test_exact_unicode_and_truncated_parameter_labels_resolve_without_dropping():
    required = ["tests/x.py::test_x", "tests/x.py::test_copy[key-💩]", "tests/x.py::test_args[-42-Argument"]
    collected = ["tests/x.py::test_x", r"tests/x.py::test_copy[key-\U0001f4a9]",
                 "tests/x.py::test_args[-42-Argument must be positive]", "tests/x.py::test_unused"]
    result = resolve_selectors(required, collected)
    assert list(result) == required and list(result.values()) == collected[:3]


@pytest.mark.parametrize("required,collected", [
    (["tests/x.py::test_args[prefix"], ["tests/x.py::test_args[prefix-one]", "tests/x.py::test_args[prefix-two]"]),
    (["tests/x.py::test_args[complete]"], ["tests/x.py::test_args[complete-longer]"]),
    (["tests/x.py::missing"], ["tests/x.py::test_x"]),
    (["tests/x.py::test_x"], ["tests/x.py::test_x", "tests/x.py::test_x"]),
    ([], ["tests/x.py::test_x"]),
])
def test_ambiguous_missing_or_broadened_selectors_fail_closed(required, collected):
    with pytest.raises(ValueError):
        resolve_selectors(required, collected)
