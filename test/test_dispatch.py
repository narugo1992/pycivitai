import hashlib
import os

import pytest

from pycivitai.client import ResourceDuplicated, ModelFoundDuplicated
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

        resource = civitai_find_online('toki/飛鳥馬トキ/时 (Blue Archive)', 'v1.4')
        assert resource.model_id == 121604
        assert resource.model_name == 'toki/飛鳥馬トキ/时 (Blue Archive)'
        assert resource.version_id == 153813
        assert resource.version_name == 'v1.4'
        assert resource.filename == 'toki_bluearchive.safetensors'
        assert resource.sha256 == '937404D0C14B2B14D87853DCCEDEC3EBF6ED0E79129E72659C2673873F6F0685'
        assert resource.is_primary
        assert resource.size == 14722160

        resource = civitai_find_online('toki/飛鳥馬トキ/时 (Blue Archive)', 'v1.4', '*.safetensors')
        assert resource.model_id == 121604
        assert resource.model_name == 'toki/飛鳥馬トキ/时 (Blue Archive)'
        assert resource.version_id == 153813
        assert resource.version_name == 'v1.4'
        assert resource.filename == 'toki_bluearchive.safetensors'
        assert resource.sha256 == '937404D0C14B2B14D87853DCCEDEC3EBF6ED0E79129E72659C2673873F6F0685'
        assert resource.is_primary
        assert resource.size == 14722160

        resource = civitai_find_online('toki/飛鳥馬トキ/时 (Blue Archive)', 'v1.4', '*.pt')
        assert resource.model_id == 121604
        assert resource.model_name == 'toki/飛鳥馬トキ/时 (Blue Archive)'
        assert resource.version_id == 153813
        assert resource.version_name == 'v1.4'
        assert resource.filename == 'toki_bluearchive.pt'
        assert resource.sha256 == '4CB82DDAE9CE0CBA475E89A083252F9179065B00BA233D0D9D8A44E323DDB53A'
        assert not resource.is_primary
        assert resource.size == 13141

        with pytest.raises(ResourceDuplicated):
            _ = civitai_find_online('toki/飛鳥馬トキ/时 (Blue Archive)', 'v1.4', '*')
        with pytest.raises(ModelFoundDuplicated):
            _ = civitai_find_online('Paimon (Genshin Impact)')

        resource = civitai_find_online('Paimon (Genshin Impact)', 'v1.4', creator='narugo1992')
        assert resource.model_id == 125187
        assert resource.model_name == 'Paimon (Genshin Impact)'
        assert resource.version_id == 156954
        assert resource.version_name == 'v1.4'
        assert resource.filename == 'paimon_genshin.safetensors'
        assert resource.sha256 == '3529E351565893CDB83DC8BDE9FF7F2E19D494B68A0AE6E43170F3C245016C8F'
        assert resource.is_primary
        assert resource.size == 14722160

    def test_civitai_download(self):
        file = civitai_download('amiya arknights (old)')
        assert os.path.getsize(file) == 25515
        assert calculate_sha256(file) == '311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0'

        file = civitai_download('明日方舟-安洁莉娜,Arknights-Angeline')
        assert os.path.getsize(file) == 37863532
        assert calculate_sha256(file) == '89A4D7CF20C76AFEE5AA4C4F28908A05D5E6BB6EB7726E5D319DE74731D09701'

    def test_civitai_find_online_by_hash(self):
        resource = civitai_find_online('FB64F545')
        assert resource.model_id == 121986
        assert resource.model_name == 'mutsuki/浅黄ムツキ/睦月 (Blue Archive)'
        assert resource.version_id == 155681
        assert resource.version_name == 'v1.4'
        assert resource.creator == 'narugo1992'
        assert resource.filename == 'mutsuki_bluearchive.pt'
        assert resource.crc32 == 'FB64F545'
        assert not resource.is_primary
        assert resource.size == 13278

        resource = civitai_find_online('B42B09FF12')
        assert resource.model_id == 6755
        assert resource.model_name == 'Cetus-Mix'
        assert resource.version_id == 78676
        assert resource.version_name == 'V4'
        assert resource.creator == 'Eagelaxis'
        assert resource.filename == 'cetusMix_v4.safetensors'
        assert resource.crc32 == '838408E0'
        assert resource.is_primary
        assert resource.size == 3894258133
