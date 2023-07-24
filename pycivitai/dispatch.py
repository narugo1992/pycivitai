import os
from functools import lru_cache
from typing import Union

from .client import find_model, find_version, find_resource, Resource
from .manager import DispatchManager


@lru_cache()
def _get_storage_dir():
    return os.environ.get('CIVITAI_HOME', os.path.expanduser('~/.cache/civitai'))


@lru_cache()
def _get_global_manager(offline: bool):
    return DispatchManager(_get_storage_dir(), offline)


def civitai_download(model: Union[str, int], version: Union[str, int, None] = None,
                     file: str = ..., offline: bool = False):
    return _get_global_manager(offline).get_file(model, version, file)


def civitai_find_online(model: Union[str, int], version: Union[str, int, None] = None, file: str = ...) -> Resource:
    model_data = find_model(model)
    version_data = find_version(model_data, version)
    return find_resource(model_data, version_data, file)
