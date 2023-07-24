import os
from unittest.mock import patch

import pytest
from hbutils.testing import isolated_directory

from pycivitai.manager import DispatchManager, LocalVersionNotFound, LocalModelNotFound
from test.testings import get_testfile


@pytest.fixture()
def sample_repo_dir():
    return get_testfile('sample_repo_1')


@pytest.fixture()
def repo_dir(sample_repo_dir):
    with isolated_directory({'repo': sample_repo_dir}):
        yield 'repo'


@pytest.fixture()
def repo_manager(repo_dir):
    yield DispatchManager(repo_dir)


@pytest.fixture()
def empty_repo_dir():
    with isolated_directory():
        os.makedirs('repo', exist_ok=True)
        yield 'repo'


@pytest.fixture()
def empty_repo_manager(empty_repo_dir):
    yield DispatchManager(empty_repo_dir)


@pytest.mark.unittest
class TestManagerDispatch:
    def test_basic(self, repo_dir, repo_manager, text_aligner):
        assert len(repo_manager.list_models()) == 1
        assert os.path.samefile(
            repo_manager.get_file('amiya arknights (old)', 'v1.0'),
            os.path.join(repo_dir, 'amiya_arknights_old__115427', 'v1_0__124870', 'files', 'amiya.pt')
        )
        assert repo_manager.total_size == 25451

        with patch('pycivitai.manager.dispatch.OFFLINE_MODE', True):
            assert os.path.samefile(
                repo_manager.get_file('amiya arknights (old)', 'v1.0'),
                os.path.join(repo_dir, 'amiya_arknights_old__115427', 'v1_0__124870', 'files', 'amiya.pt')
            )
            assert os.path.samefile(
                repo_manager.get_file('amiya arknights (old)'),
                os.path.join(repo_dir, 'amiya_arknights_old__115427', 'v1_0__124870', 'files', 'amiya.pt')
            )
            with pytest.raises(LocalVersionNotFound):
                _ = repo_manager.get_file('amiya arknights (old)', 'v1.1')
            with pytest.raises(LocalModelNotFound):
                _ = repo_manager.get_file('明日方舟-安洁莉娜,Arknights-Angeline')

        assert os.path.samefile(
            repo_manager.get_file('amiya arknights (old)'),
            os.path.join(repo_dir, 'amiya_arknights_old__115427', 'v1_1__124885', 'files', 'amiya.pt')
        )
        assert os.path.samefile(
            repo_manager.get_file('amiya arknights (old)', 'v1.1'),
            os.path.join(repo_dir, 'amiya_arknights_old__115427', 'v1_1__124885', 'files', 'amiya.pt')
        )
        assert os.path.samefile(
            repo_manager.get_file('明日方舟-安洁莉娜,Arknights-Angeline'),
            os.path.join(repo_dir, '明日方舟_安洁莉娜_arknights_angeline__5632', 'v1_0__6555', 'files',
                         '安洁莉娜5(头巾headband,三套衣服coat,multiple strap,swimsuit),.safetensors')
        )
        assert repo_manager.total_size == 37914498

        assert repr(repo_manager) == '<DispatchManager directory: \'repo\'>'
        text_aligner.assert_equal(
            """
<DispatchManager directory: 'repo'>
├── <ModelManager model: '明日方舟_安洁莉娜_arknights_angeline'>
│   └── <VersionManager model: '明日方舟_安洁莉娜_arknights_angeline', version: 'v1_0'>
│       └── LocalFile(filename='安洁莉娜5(头巾headband,三套衣服coat,multiple strap,swimsuit),.safetensors', hash='89A4D7CF20C76AFEE5AA4C4F28908A05D5E6BB6EB7726E5D319DE74731D09701', size=37863532, is_primary=True)
└── <ModelManager model: 'amiya_arknights_old'>
    ├── <VersionManager model: 'amiya_arknights_old', version: 'v1_0'>
    │   └── LocalFile(filename='amiya.pt', hash='259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED', size=25451, is_primary=True)
    └── <VersionManager model: 'amiya_arknights_old', version: 'v1_1'>
        └── LocalFile(filename='amiya.pt', hash='311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0', size=25515, is_primary=True)
            """,
            str(repo_manager),
        )

        repo_manager.delete_version('amiya arknights (old)', 'v1.0')
        assert repo_manager.total_size == 37889047
        text_aligner.assert_equal(
            """
<DispatchManager directory: 'repo'>
├── <ModelManager model: '明日方舟_安洁莉娜_arknights_angeline'>
│   └── <VersionManager model: '明日方舟_安洁莉娜_arknights_angeline', version: 'v1_0'>
│       └── LocalFile(filename='安洁莉娜5(头巾headband,三套衣服coat,multiple strap,swimsuit),.safetensors', hash='89A4D7CF20C76AFEE5AA4C4F28908A05D5E6BB6EB7726E5D319DE74731D09701', size=37863532, is_primary=True)
└── <ModelManager model: 'amiya_arknights_old'>
    └── <VersionManager model: 'amiya_arknights_old', version: 'v1_1'>
        └── LocalFile(filename='amiya.pt', hash='311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0', size=25515, is_primary=True)
            """,
            str(repo_manager),
        )

        repo_manager.delete_model('明日方舟_安洁莉娜_arknights_angeline')
        assert repo_manager.total_size == 25515
        text_aligner.assert_equal(
            """
<DispatchManager directory: 'repo'>
└── <ModelManager model: 'amiya_arknights_old'>
    └── <VersionManager model: 'amiya_arknights_old', version: 'v1_1'>
        └── LocalFile(filename='amiya.pt', hash='311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0', size=25515, is_primary=True)
            """,
            str(repo_manager),
        )
