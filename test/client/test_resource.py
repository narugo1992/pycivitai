import pytest

from pycivitai.client import find_model, ModelNotFound, find_version, ModelVersionNotFound, find_resource, \
    ResourceNotFound, ResourceDuplicated


@pytest.fixture(scope='module')
def amiya_pt_name():
    return 'amiya arknights (old)'


@pytest.fixture(scope='module')
def amiya_pt_id():
    return 115427


@pytest.fixture(scope='module')
def amiya_model_data(amiya_pt_id):
    return find_model(amiya_pt_id)


@pytest.fixture(scope='module')
def amiya_model_v1_0_data(amiya_model_data):
    return find_version(amiya_model_data, 'v1.0')


@pytest.fixture(scope='module')
def amiya_model_v1_1_data(amiya_model_data):
    return find_version(amiya_model_data, 'v1.1')


@pytest.mark.unittest
class TestClientResource:
    def test_find_model_by_name(self, amiya_pt_name, amiya_pt_id):
        data = find_model(amiya_pt_name)
        assert data, f'Data should be non-empty, but {data!r} found.'
        assert data['id'] == amiya_pt_id
        assert data['name'] == amiya_pt_name

    def test_find_model_by_id(self, amiya_pt_name, amiya_pt_id):
        data = find_model(amiya_pt_id)
        assert data, f'Data should be non-empty, but {data!r} found.'
        assert data['id'] == amiya_pt_id
        assert data['name'] == amiya_pt_name

    def test_find_model_by_id_str(self, amiya_pt_name, amiya_pt_id):
        data = find_model(str(amiya_pt_id))
        assert data, f'Data should be non-empty, but {data!r} found.'
        assert data['id'] == amiya_pt_id
        assert data['name'] == amiya_pt_name

    def test_find_non_exist_model(self):
        with pytest.raises(ModelNotFound):
            _ = find_model(-1)
        with pytest.raises(ModelNotFound):
            _ = find_model('model_not_found_' * 10)

    def test_find_model_invalid(self):
        with pytest.raises(TypeError):
            _ = find_model(None)

    def test_find_version(self, amiya_model_data):
        data = find_version(amiya_model_data, 'v1.0')
        assert data['id'] == 124870
        assert data['name'] == 'v1.0'

        data = find_version(amiya_model_data, 'v1.1')
        assert data['id'] == 124885
        assert data['name'] == 'v1.1'

        data = find_version(amiya_model_data)
        assert data['id'] == 124885
        assert data['name'] == 'v1.1'

        with pytest.raises(ModelVersionNotFound):
            _ = find_version(amiya_model_data, 'v100.0')

    def test_find_resource(self, amiya_model_data, amiya_model_v1_0_data, amiya_model_v1_1_data,
                           amiya_pt_id, amiya_pt_name):
        data = find_resource(amiya_model_data, amiya_model_v1_0_data)
        assert data.model_id == amiya_pt_id
        assert data.model_name == amiya_pt_name
        assert data.version_id == 124870
        assert data.version_name == 'v1.0'
        assert data.filename == 'amiya.pt'
        assert data.sha256 == '259BE5CF344CDBCA981B389BE7C105B8993D9D340C172C556F2E0E8283E3DBED'
        assert data.size == 25451

        data = find_resource(amiya_model_data, amiya_model_v1_1_data)
        assert data.model_id == amiya_pt_id
        assert data.model_name == amiya_pt_name
        assert data.version_id == 124885
        assert data.version_name == 'v1.1'
        assert data.filename == 'amiya.pt'
        assert data.sha256 == '311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0'
        assert data.size == 25515

        data = find_resource(amiya_model_data, amiya_model_v1_1_data, '*.pt')
        assert data.model_id == amiya_pt_id
        assert data.model_name == amiya_pt_name
        assert data.version_id == 124885
        assert data.version_name == 'v1.1'
        assert data.filename == 'amiya.pt'
        assert data.sha256 == '311279B35743D703DED68DF6ECC9F1E850A9F6976494CC8AF434DF3AA0E238A0'
        assert data.size == 25515

        with pytest.raises(ResourceNotFound):
            _ = find_resource(amiya_model_data, amiya_model_v1_1_data, '*.ckpt')
        with pytest.raises(TypeError):
            _ = find_resource(amiya_model_data, amiya_model_v1_1_data, 233)

    def test_with_Cetus_Mix(self):
        model_data = find_model('Cetus-Mix')
        version_data = find_version(model_data, 'V4')

        data = find_resource(model_data, version_data)
        assert data.model_id == 6755
        assert data.model_name == 'Cetus-Mix'
        assert data.version_id == 78676
        assert data.version_name == 'V4'
        assert data.filename == 'cetusMix_v4.safetensors'
        assert data.sha256 == 'B42B09FF12CA9CD70D78AA8210F8D4577EC513FC1484A68615385B8076292639'
        assert data.size == 3894258133

        data = find_resource(model_data, version_data, '*.safetensors')
        assert data.model_id == 6755
        assert data.model_name == 'Cetus-Mix'
        assert data.version_id == 78676
        assert data.version_name == 'V4'
        assert data.filename == 'cetusMix_v4.safetensors'
        assert data.sha256 == 'B42B09FF12CA9CD70D78AA8210F8D4577EC513FC1484A68615385B8076292639'
        assert data.size == 3894258133

        data = find_resource(model_data, version_data, '*.vae.pt')
        assert data.model_id == 6755
        assert data.model_name == 'Cetus-Mix'
        assert data.version_id == 78676
        assert data.version_name == 'V4'
        assert data.filename == 'MoistMix.vae.pt'
        assert data.sha256 == 'DF3C506E51B7EE1D7B5A6A2BB7142D47D488743C96AA778AFB0F53A2CDC2D38D'
        assert data.size == 404662241

        with pytest.raises(ResourceNotFound):
            _ = find_resource(model_data, version_data, '*.ckpt')
        with pytest.raises(ResourceDuplicated):
            _ = find_resource(model_data, version_data, '*')
