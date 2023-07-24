import os
from functools import lru_cache
from typing import Union

from .manager import DispatchManager


@lru_cache()
def _get_storage_dir():
    return os.environ.get('CIVITAI_HOME', os.path.expanduser('~/.cache/civitai'))


@lru_cache()
def _get_global_manager(offline: bool):
    return DispatchManager(_get_storage_dir(), offline)


def civitiai_download(model: Union[str, int], version: Union[str, int, None] = None,
                      file: str = ..., offline: bool = False):
    return _get_global_manager(offline).get_file(model, version, file)
