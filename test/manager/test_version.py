import os.path

import pytest
from hbutils.testing import isolated_directory

from pycivitai.client import ResourceNotFound
from pycivitai.manager import VersionManager, LocalFileNotFound, LocalFile
from ..testings import get_testfile


@pytest.fixture()
def amiya_v1_0_version_dir():
    with isolated_directory({'version': get_testfile('amiya_v1_0')}):
        yield 'version'


@pytest.fixture()
def amiya_v1_0_version_manager(amiya_v1_0_version_dir):
    yield VersionManager(amiya_v1_0_version_dir, 115427, 124870)


@pytest.fixture()
def amiya_v1_0_version_size_dir():
    with isolated_directory({'version': get_testfile('amiya_v1_0')}):
        with open(os.path.join('version', 'files', 'amiya.pt'), 'w') as f:
            f.write('')

        assert os.path.getsize(os.path.join('version', 'files', 'amiya.pt')) == 0
        yield 'version'


@pytest.fixture()
def amiya_v1_0_version_size_manager(amiya_v1_0_version_size_dir):
    yield VersionManager(amiya_v1_0_version_size_dir, 115427, 124870)


@pytest.fixture()
def amiya_v1_0_version_np_dir():
    with isolated_directory({'version': get_testfile('amiya_v1_0')}):
        with open(os.path.join('version', 'files', 'amiya.pt'), 'w') as f:
            f.write('')

        os.remove(os.path.join('version', 'primary'))
        yield 'version'


@pytest.fixture()
def amiya_v1_0_version_np_manager(amiya_v1_0_version_np_dir):
    yield VersionManager(amiya_v1_0_version_np_dir, 115427, 124870)


@pytest.mark.unittest
class TestManagerVersion:
    def test_exist(self, amiya_v1_0_version_manager: VersionManager, text_aligner):
        assert amiya_v1_0_version_manager.list_files() == [
            LocalFile('amiya.pt', '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', 25451, True)
        ]
        assert amiya_v1_0_version_manager.total_size == 25451
        assert os.path.samefile(
            amiya_v1_0_version_manager.get_file(),
            os.path.join('version', 'files', 'amiya.pt')
        )
        with pytest.raises(ResourceNotFound):
            assert amiya_v1_0_version_manager.get_file('amiya.ckpt')

        assert repr(amiya_v1_0_version_manager) == '<VersionManager model: 115427, version: 124870>'
        text_aligner.assert_equal(
            """
<VersionManager model: 115427, version: 124870>
└── LocalFile(filename='amiya.pt', hash='259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', size=25451, is_primary=True)
            """,
            str(amiya_v1_0_version_manager)
        )
        with pytest.raises(LocalFileNotFound):
            amiya_v1_0_version_manager.delete_file('amiya.ckpt')
        amiya_v1_0_version_manager.delete_file('amiya.pt')
        assert amiya_v1_0_version_manager.list_files() == []
        assert not os.path.exists(os.path.join('version', 'files', 'amiya.pt'))

    def test_size_not_match(self, amiya_v1_0_version_size_manager: VersionManager, text_aligner):
        assert amiya_v1_0_version_size_manager.list_files() == [
            LocalFile('amiya.pt', '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', 0, True)
        ]
        assert amiya_v1_0_version_size_manager.total_size == 0
        assert os.path.samefile(
            amiya_v1_0_version_size_manager.get_file(),
            os.path.join('version', 'files', 'amiya.pt')
        )
        assert os.path.samefile(
            amiya_v1_0_version_size_manager.get_file('*.pt'),
            os.path.join('version', 'files', 'amiya.pt')
        )
        assert amiya_v1_0_version_size_manager.list_files() == [
            LocalFile('amiya.pt', '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', 25451, True)
        ]
        assert amiya_v1_0_version_size_manager.total_size == 25451

        assert repr(amiya_v1_0_version_size_manager) == '<VersionManager model: 115427, version: 124870>'
        text_aligner.assert_equal(
            """
<VersionManager model: 115427, version: 124870>
└── LocalFile(filename='amiya.pt', hash='259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', size=25451, is_primary=True)
            """,
            str(amiya_v1_0_version_size_manager)
        )
        with pytest.raises(LocalFileNotFound):
            amiya_v1_0_version_size_manager.delete_file('amiya.ckpt')
        amiya_v1_0_version_size_manager.delete_file('amiya.pt')
        assert amiya_v1_0_version_size_manager.list_files() == []
        assert not os.path.exists(os.path.join('version', 'files', 'amiya.pt'))

    def test_exist_no_primary(self, amiya_v1_0_version_np_manager: VersionManager, text_aligner):
        assert amiya_v1_0_version_np_manager.list_files() == [
            LocalFile('amiya.pt', '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', 0, False)
        ]
        assert amiya_v1_0_version_np_manager.total_size == 0
        assert os.path.samefile(
            amiya_v1_0_version_np_manager.get_file('*.pt'),
            os.path.join('version', 'files', 'amiya.pt')
        )
        assert amiya_v1_0_version_np_manager.list_files() == [
            LocalFile('amiya.pt', '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', 25451, True)
        ]
        assert amiya_v1_0_version_np_manager.total_size == 25451
        with pytest.raises(ResourceNotFound):
            assert amiya_v1_0_version_np_manager.get_file('amiya.ckpt')

        assert repr(amiya_v1_0_version_np_manager) == '<VersionManager model: 115427, version: 124870>'
        text_aligner.assert_equal(
            """
<VersionManager model: 115427, version: 124870>
└── LocalFile(filename='amiya.pt', hash='259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', size=25451, is_primary=True)
            """,
            str(amiya_v1_0_version_np_manager)
        )
        with pytest.raises(LocalFileNotFound):
            amiya_v1_0_version_np_manager.delete_file('amiya.ckpt')
        amiya_v1_0_version_np_manager.delete_file('amiya.pt')
        assert amiya_v1_0_version_np_manager.list_files() == []
        assert not os.path.exists(os.path.join('version', 'files', 'amiya.pt'))
