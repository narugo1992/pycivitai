import pytest
from hbutils.testing import TextAligner


@pytest.fixture()
def text_aligner():
    return TextAligner().multiple_lines()
