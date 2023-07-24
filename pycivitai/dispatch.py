import os
from functools import lru_cache
from typing import Union

from .client import find_model, find_version, find_resource, Resource
from .manager import DispatchManager


@lru_cache()
def _get_storage_dir():
    """
    Get the directory path where the downloaded models will be managed.

    :return: The storage directory path.
    :rtype: str
    """
    return os.environ.get('CIVITAI_HOME', os.path.expanduser('~/.cache/civitai'))


@lru_cache()
def _get_global_manager(offline: bool):
    """
    Get the global manager responsible for managing all downloaded models.

    :param offline: If True, the manager operates in offline mode, using locally downloaded resources.
    :return: The global DispatchManager instance.
    :rtype: DispatchManager
    """
    return DispatchManager(_get_storage_dir(), offline)


def civitai_download(model: Union[str, int], version: Union[str, int, None] = None,
                     file: str = None, offline: bool = False):
    """
    Download and get the local file path of the specified model file.

    :param model: The name or ID of the model to download and manage.
    :type model: Union[str, int]
    :param version: The version ID or name of the model version. If None, the latest version is used.
    :type version: Union[str, int, None]
    :param file: The pattern or name of the file to get. If None, the primary file will be returned.
    :type file: str
    :param offline: If True, the manager operates in offline mode, using locally downloaded resources.
    :type offline: bool
    :return: The local path of the specified model file.
    :rtype: str
    """
    return _get_global_manager(offline).get_file(model, version, file)


def civitai_find_online(model: Union[str, int], version: Union[str, int, None] = None, file: str = None) -> Resource:
    """
    Find the online model resource (file) information from civitai.com.

    :param model: The name or ID of the model.
    :type model: Union[str, int]
    :param version: The version ID or name of the model version. If None, the latest version is used.
    :type version: Union[str, int, None]
    :param file: The pattern or name of the file to find. If None, the primary file will be returned.
    :type file: str
    :return: The Resource object containing the information about the specified model file.
    :rtype: Resource
    """
    model_data = find_model(model)
    version_data = find_version(model_data, version)
    return find_resource(model_data, version_data, file)
