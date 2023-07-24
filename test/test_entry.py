from unittest.mock import patch

import pytest
from hbutils.testing import simulate_entry, isolated_directory

from pycivitai.config.meta import __TITLE__, __VERSION__
from pycivitai.entry import cli
from pycivitai.manager import DispatchManager
from .testings import get_testfile

try:
    import InquirerPy
except (ModuleNotFoundError, ImportError):
    InquirerPy = None


@pytest.fixture()
def sample_repo_dir():
    return get_testfile('sample_repo_1')


@pytest.fixture()
def repo_dir(sample_repo_dir):
    with isolated_directory({'repo': sample_repo_dir}):
        def _get_manager(offline):
            return DispatchManager('repo', offline)

        with patch('pycivitai.entry._get_global_manager', _get_manager):
            yield


@pytest.mark.unittest
class TestEntry:
    def test_version(self):
        result = simulate_entry(cli, ['cli', '-v'])
        assert result.exitcode == 0
        assert __TITLE__.lower() in result.stdout.lower()
        assert __VERSION__.lower() in result.stdout.lower()

    def test_delete_all(self, repo_dir, text_aligner):
        result = simulate_entry(cli, ['cli', 'delete-cache', '-A'])
        assert result.exitcode == 0
        text_aligner.assert_equal(
            'All models deleted, total 24.854 KiB.',
            result.stdout,
        )
