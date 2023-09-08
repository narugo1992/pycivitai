import os.path
from unittest.mock import patch

import pytest
from hbutils.testing import isolated_directory

from pycivitai.manager import ModelManager, LocalVersionNotFound
from test.testings import get_testfile


@pytest.fixture()
def sample_model_dir():
    return get_testfile('sample_repo_1', 'amiya_arknights_old__narugo1992__115427')


@pytest.fixture()
def amiya_model_dir(sample_model_dir):
    with isolated_directory({'model': sample_model_dir}):
        yield 'model'


@pytest.fixture()
def amiya_model_manager(amiya_model_dir):
    yield ModelManager(amiya_model_dir, 115427)


@pytest.mark.unittest
class TestManagerModel:
    def test_default(self, amiya_model_manager: ModelManager, amiya_model_dir, text_aligner):
        vms = amiya_model_manager.list_versions()
        assert len(vms) == 1
        assert vms[0].version == 'v1_0'
        assert vms[0].model_name_or_id == 115427
        assert amiya_model_manager.total_size == 25451

        assert os.path.samefile(
            amiya_model_manager.get_file('v1.0'),
            os.path.join(amiya_model_dir, 'v1_0__124870', 'files', 'amiya.pt'),
        )
        assert os.path.samefile(
            amiya_model_manager.get_file(),
            os.path.join(amiya_model_dir, 'v1_1__124885', 'files', 'amiya.pt'),
        )

        vms = amiya_model_manager.list_versions()
        assert len(vms) == 2
        vms = {vm.version: vm for vm in vms}
        assert vms['v1_0'].version == 'v1_0'
        assert vms['v1_0'].model_name_or_id == 115427
        assert vms['v1_1'].version == 'v1_1'
        assert vms['v1_1'].model_name_or_id == 115427
        assert amiya_model_manager.total_size == 50966

        with patch('pycivitai.manager.model.OFFLINE_MODE', True):
            assert os.path.samefile(
                amiya_model_manager.get_file('v1.0'),
                os.path.join(amiya_model_dir, 'v1_0__124870', 'files', 'amiya.pt'),
            )
            assert os.path.samefile(
                amiya_model_manager.get_file(),
                os.path.join(amiya_model_dir, 'v1_1__124885', 'files', 'amiya.pt'),
            )
            with pytest.raises(LocalVersionNotFound):
                amiya_model_manager.get_file('v100.0')

        assert repr(amiya_model_manager) == '<ModelManager model: 115427>'
        text_aligner.assert_equal(
            """
<ModelManager model: 115427>
├── <VersionManager model: 115427, version: 'v1_0'>
│   └── LocalFile(filename='amiya.pt', hash='259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', size=25451, is_primary=True)
└── <VersionManager model: 115427, version: 'v1_1'>
    └── LocalFile(filename='amiya.pt', hash='311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0', size=25515, is_primary=True)
            """,
            str(amiya_model_manager),
        )

        amiya_model_manager.delete_version('v1.0')
        assert len(amiya_model_manager.list_versions()) == 1
        text_aligner.assert_equal(
            """
<ModelManager model: 115427>
└── <VersionManager model: 115427, version: 'v1_1'>
    └── LocalFile(filename='amiya.pt', hash='311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0', size=25515, is_primary=True)
            """,
            str(amiya_model_manager),
        )

        amiya_model_manager.delete_version(124885)
        assert len(amiya_model_manager.list_versions()) == 0

        with pytest.raises(LocalVersionNotFound):
            amiya_model_manager.delete_version('v100.0')
