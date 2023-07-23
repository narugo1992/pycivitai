import os
from typing import Iterator, Tuple, Union

import requests

from .base import _soft_name_strip
from .model import ModelManager
from ..client import find_model, OFFLINE_MODE, OfflineModeEnabled


class LocalModelNotFound(Exception):
    pass


class LocalModelDuplicated(Exception):
    pass


class DispatchManager:
    def __init__(self, root_dir: str, offline: bool = False):
        self.root_dir = root_dir

        os.makedirs(self.root_dir, exist_ok=True)
        self._offline = offline

    def _model_path(self, model_name: str, model_id: int):
        return os.path.join(self.root_dir, f'{_soft_name_strip(model_name)}__{model_id}')

    def _list_local_models(self) -> Iterator[Tuple[str, int, str]]:
        for dir_ in os.listdir(self.root_dir):
            segs = dir_.split('__')
            if os.path.isdir(os.path.join(self.root_dir, dir_)) and len(segs) == 2:
                model_name, model_id = segs
                model_id = int(model_id)
                yield model_name, model_id, os.path.join(self.root_dir, dir_)

    def _find_online_model(self, model_name_or_id: Union[str, int]):
        model_data = find_model(model_name_or_id)
        model_id, model_name = model_data['id'], model_data['name']
        return model_name, model_id, self._model_path(model_name, model_id), model_data

    def _find_local_model(self, model_name_or_id: Union[str, int]):
        valid_models = []
        for model_name, model_id, model_dir in self._list_local_models():
            if (_soft_name_strip(str(model_name_or_id)) == model_name) or (model_id == model_name_or_id):
                valid_models.append((model_name, model_id, model_dir))

        if not valid_models:
            raise LocalModelNotFound(model_name_or_id)
        elif len(valid_models) > 1:
            raise LocalModelDuplicated(valid_models)
        else:
            model_name, model_id, model_dir = valid_models[0]
            return model_name, model_id, model_dir

    def _get_model_manager(self, model_name_or_id: Union[str, int, None]):
        try:
            if OFFLINE_MODE or self._offline:
                raise OfflineModeEnabled

            model_name, model_id, model_dir, model_data = self._find_online_model(model_name_or_id)
            return ModelManager(model_dir, model_name_or_id, model_data, offline=False)
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OfflineModeEnabled,
        ):
            model_name, model_id, model_dir = self._find_local_model(model_name_or_id)
            return ModelManager(model_dir, model_name_or_id, offline=True)

    def get_file(self, model_name_or_id: Union[str, int, None],
                 version: Union[str, int, None] = None, pattern: str = ...):
        return self._get_model_manager(model_name_or_id).get_file(version, pattern)
