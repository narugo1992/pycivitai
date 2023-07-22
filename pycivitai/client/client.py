from ..constant import ENDPOINT
from ..utils import get_requests_session, srequest


class CivitAIClient:
    def __init__(self, session=None):
        self.session = session or get_requests_session()

    def model_info_by_id(self, model_id):
        resp = srequest(self.session, 'GET', f'{ENDPOINT}/api/v1/models/{model_id}')
        return resp.json()
