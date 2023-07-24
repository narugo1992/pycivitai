import hashlib
import os

import pytest

from pycivitai.client import ResourceDuplicated
from pycivitai.dispatch import civitai_find_online, civitai_download


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    block_size = 65536

    with open(file_path, 'rb') as file:
        while True:
            data = file.read(block_size)
            if not data:
                break
            sha256_hash.update(data)

    return sha256_hash.hexdigest().upper()


@pytest.mark.unittest
class TestDispatch:
    def test_civitai_find_online(self):
        resource = civitai_find_online('明日方舟-安洁莉娜,Arknights-Angeline')
        assert resource.model_id == 5632
        assert resource.model_name == '明日方舟-安洁莉娜,Arknights-Angeline'
        assert resource.version_id == 6555
        assert resource.version_name == 'v1.0'
        assert resource.filename == '安洁莉娜5(头巾headband,三套衣服coat,multiple strap,swimsuit),.safetensors'
        assert resource.sha256 == '89A4D7CF20C76AFEE5AA4C4F28908A05D5E6BB6EB7726E5D319DE74731D09701'
        assert resource.is_primary
        assert resource.size == 37863532

        resource = civitai_find_online('Cetus-Mix', 'V4')
        assert resource.model_id == 6755
        assert resource.model_name == 'Cetus-Mix'
        assert resource.version_id == 78676
        assert resource.version_name == 'V4'
        assert resource.filename == 'cetusMix_v4.safetensors'
        assert resource.sha256 == 'B42B09FF12CA9CD70D78AA8210F8D4577EC513FC1484A68615385B8076292639'
        assert resource.is_primary
        assert resource.size == 3894258133

        resource = civitai_find_online('Cetus-Mix', 'V4', '*.safetensors')
        assert resource.model_id == 6755
        assert resource.model_name == 'Cetus-Mix'
        assert resource.version_id == 78676
        assert resource.version_name == 'V4'
        assert resource.filename == 'cetusMix_v4.safetensors'
        assert resource.sha256 == 'B42B09FF12CA9CD70D78AA8210F8D4577EC513FC1484A68615385B8076292639'
        assert resource.is_primary
        assert resource.size == 3894258133

        resource = civitai_find_online('Cetus-Mix', 'V4', '*.vae.pt')
        assert resource.model_id == 6755
        assert resource.model_name == 'Cetus-Mix'
        assert resource.version_id == 78676
        assert resource.version_name == 'V4'
        assert resource.filename == 'MoistMix.vae.pt'
        assert resource.sha256 == 'DF3C506E51B7EE1D7B5A6A2BB7142D47D488743C96AA778AFB0F53A2CDC2D38D'
        assert not resource.is_primary
        assert resource.size == 404662241

        with pytest.raises(ResourceDuplicated):
            _ = civitai_find_online('Cetus-Mix', 'V4', '*')

    def test_civitai_download(self):
        file = civitai_download('amiya arknights (old)')
        assert os.path.getsize(file) == 25515
        assert calculate_sha256(file) == '311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0'

        file = civitai_download('明日方舟-安洁莉娜,Arknights-Angeline')
        assert os.path.getsize(file) == 37863532
        assert calculate_sha256(file) == '89A4D7CF20C76AFEE5AA4C4F28908A05D5E6BB6EB7726E5D319DE74731D09701'
