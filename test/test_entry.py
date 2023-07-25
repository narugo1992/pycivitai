import os
from unittest import skipUnless
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
def sample_repo(sample_repo_dir):
    with isolated_directory({'repo': sample_repo_dir}):
        def _get_manager(offline):
            return DispatchManager('repo', offline)

        with patch('pycivitai.entry._get_global_manager', _get_manager), \
                patch('pycivitai.dispatch._get_global_manager', _get_manager):
            yield DispatchManager('repo', offline=True)


@pytest.fixture()
def empty_repo():
    with isolated_directory():
        os.makedirs('repo', exist_ok=True)

        def _get_manager(offline):
            return DispatchManager('repo', offline)

        with patch('pycivitai.entry._get_global_manager', _get_manager), \
                patch('pycivitai.dispatch._get_global_manager', _get_manager):
            yield DispatchManager('repo', offline=True)


@pytest.fixture()
def mock_all_choices():
    from InquirerPy.base import Choice
    def _get_choices(choices):
        retval = []
        for item in choices:
            if isinstance(item, Choice):
                retval.append(item.value)

        return retval

    with patch('pycivitai.entry._get_choices', _get_choices):
        yield


@pytest.fixture()
def mock_no_choices():
    def _get_choices(choices):
        return []

    with patch('pycivitai.entry._get_choices', _get_choices):
        yield


@pytest.mark.unittest
class TestEntry:
    def test_version(self):
        result = simulate_entry(cli, ['cli', '-v'])
        assert result.exitcode == 0
        assert __TITLE__.lower() in result.stdout.lower()
        assert __VERSION__.lower() in result.stdout.lower()

    def test_delete_all(self, sample_repo, text_aligner):
        result = simulate_entry(cli, ['cli', 'delete-cache', '-A'])
        assert result.exitcode == 0
        text_aligner.assert_equal(
            'All models deleted, total 24.854 KiB.',
            result.stdout,
        )

    @skipUnless(InquirerPy is not None, 'InquirePy required')
    def test_delete_with_tui(self, sample_repo, text_aligner, mock_all_choices):
        assert sample_repo.total_size == 25451
        with patch('pycivitai.entry._confirm', lambda x: True):
            result = simulate_entry(cli, ['cli', 'delete-cache'])
            assert result.exitcode == 0
            text_aligner.assert_equal(
                'Deletion complete!',
                result.stdout,
            )
        assert sample_repo.total_size == 0

    @skipUnless(InquirerPy is not None, 'InquirePy required')
    def test_delete_with_tui_cancelled(self, sample_repo, text_aligner, mock_all_choices):
        assert sample_repo.total_size == 25451
        with patch('pycivitai.entry._confirm', lambda x: False):
            result = simulate_entry(cli, ['cli', 'delete-cache'])
            assert result.exitcode == 0
        assert sample_repo.total_size == 25451

    @skipUnless(InquirerPy is not None, 'InquirePy required')
    def test_delete_with_tui_nothing(self, empty_repo, text_aligner, mock_all_choices):
        assert empty_repo.total_size == 0
        with patch('pycivitai.entry._confirm', lambda x: True):
            result = simulate_entry(cli, ['cli', 'delete-cache'])
            assert result.exitcode == 0
            text_aligner.assert_equal(
                'No models found to delete.',
                result.stdout,
            )
        assert empty_repo.total_size == 0

    @skipUnless(InquirerPy is not None, 'InquirePy required')
    def test_delete_with_tui_no_choice(self, sample_repo, text_aligner, mock_no_choices):
        assert sample_repo.total_size == 25451
        with patch('pycivitai.entry._confirm', lambda x: True):
            result = simulate_entry(cli, ['cli', 'delete-cache'])
            assert result.exitcode == 0
            text_aligner.assert_equal(
                'Deletion cancelled.',
                result.stdout,
            )
        assert sample_repo.total_size == 25451

    def test_get(self, sample_repo):
        result = simulate_entry(cli, ['cli', 'get', '-m', 'amiya arknights (old)'])
        assert result.exitcode == 0
        assert os.path.samefile(
            result.stdout.strip(),
            os.path.join('repo', 'amiya_arknights_old__115427', 'v1_1__124885', 'files', 'amiya.pt'),
        )

    def test_get_offline(self, sample_repo):
        result = simulate_entry(cli, ['cli', 'get', '-m', 'amiya arknights (old)', '--offline'])
        assert result.exitcode == 0
        assert os.path.samefile(
            result.stdout.strip(),
            os.path.join('repo', 'amiya_arknights_old__115427', 'v1_0__124870', 'files', 'amiya.pt'),
        )

    def test_get_specific_version(self, sample_repo):
        result = simulate_entry(cli, ['cli', 'get', '-m', 'amiya arknights (old)', '-v', 'v1.0'])
        assert result.exitcode == 0
        assert os.path.samefile(
            result.stdout.strip(),
            os.path.join('repo', 'amiya_arknights_old__115427', 'v1_0__124870', 'files', 'amiya.pt'),
        )
