import pytest
import requests

from pycivitai.client import configure_http_backend, get_session


@pytest.fixture()
def session_getter():
    def _func():
        session = requests.session()
        session.custom_prop = 1
        return session

    return _func


@pytest.mark.unittest
class TestClientHttp:
    def test_configure_http_backend(self, session_getter):
        try:
            configure_http_backend(session_getter)
            session = get_session()
            assert session.custom_prop == 1
        finally:
            configure_http_backend(requests.session)
            session = get_session()
            assert not hasattr(session, 'custom_prop')
