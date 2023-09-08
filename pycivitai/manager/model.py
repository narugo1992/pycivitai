import os
import shutil
from typing import Union, Optional, Iterator, Tuple, List

import requests
from filelock import FileLock
from hbutils.collection import nested_map
from hbutils.string import format_tree

from .base import _soft_name_strip
from .version import VersionManager
from ..client import find_model, find_version, OFFLINE_MODE, OfflineModeEnabled


class LocalVersionNotFound(Exception):
    pass


class LocalVersionDuplicated(Exception):
    pass


class ModelManager:
    """
    Management of specific model.
    """

    def __init__(self, root_dir: str, model_name_or_id: Union[str, int], creator: Optional[str] = None,
                 model_data: Optional[dict] = None, offline: bool = False):
        """
        Manages multiple versions of a model downloaded from civitai.com.

        :param root_dir: The root directory where the model versions will be managed.
        :param model_name_or_id: The name or ID of the model to manage.
        :param creator: Name of creator. ``None`` means anyone.
        :param model_data: Optional dictionary containing model information to avoid fetching it from the API.
        :param offline: If True, the manager operates in offline mode, using locally downloaded resources.
        """
        self.root_dir = root_dir
        self.model_name_or_id = model_name_or_id
        self.creator = creator
        self._model_data = model_data

        os.makedirs(root_dir, exist_ok=True)
        self._d_versions = os.path.join(self.root_dir)
        self._f_lock = os.path.join(self.root_dir, '.filelock')
        self.lock = FileLock(self._f_lock)
        self._offline = offline

    def _get_model(self):
        if not self._model_data:
            self._model_data = find_model(self.model_name_or_id, self.creator)
        return self._model_data

    def _version_path(self, version_name: str, version_id: int):
        return os.path.join(self._d_versions, f'{_soft_name_strip(version_name)}__{version_id}')

    def _list_local_versions(self) -> Iterator[Tuple[str, int, str]]:
        for dir_ in os.listdir(self._d_versions):
            segs = dir_.split('__')
            if os.path.isdir(os.path.join(self._d_versions, dir_)) and len(segs) == 2:
                version_name, version_id = segs
                version_id = int(version_id)
                yield version_name, version_id, os.path.join(self._d_versions, dir_)

    def _find_online_version(self, version: Union[str, int, None]):
        version_data = find_version(self._get_model(), version)
        version_id, version_name = version_data['id'], version_data['name']
        return version_name, version_id, self._version_path(version_name, version_id)

    def _find_local_version(self, version: Union[str, int, None]):
        valid_versions = []
        for version_name, version_id, version_dir in self._list_local_versions():
            if (version is None) or ((version is not None) and (
                    (_soft_name_strip(str(version)) == version_name) or
                    (version_id == version)
            )):
                valid_versions.append((version_name, version_id, version_dir))

        if not valid_versions:
            raise LocalVersionNotFound(self.model_name_or_id, version)
        else:
            if version is None:
                version_name, version_id, version_dir = sorted(valid_versions, key=lambda x: -x[1])[0]
            else:
                if len(valid_versions) > 1:
                    raise LocalVersionDuplicated(self.model_name_or_id, valid_versions)
                else:
                    version_name, version_id, version_dir = valid_versions[0]

            return version_name, version_id, version_dir

    def _get_version_manager(self, version: Union[str, int, None]):
        try:
            if OFFLINE_MODE or self._offline:
                raise OfflineModeEnabled

            version_name, version_id, version_dir = self._find_online_version(version)
            return VersionManager(
                version_dir, self.model_name_or_id, self.creator, version_name,
                model_data=self._get_model(), offline=False
            )

        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OfflineModeEnabled,
        ):
            version_name, version_id, version_dir = self._find_local_version(version)
            return VersionManager(version_dir, self.model_name_or_id, self.creator, version_name, offline=True)

    def get_file(self, version: Union[str, int, None] = None, pattern: str = None):
        """
        Get the local file path of the specified model file.

        :param version: The version ID or name of the model version. If None, the latest version is used.
        :type version: Union[str, int, None]
        :param pattern: The pattern or name of the file to get. If None, the primary file will be returned.
        :type pattern: str
        :return: The local path of the specified model file.
        :rtype: str
        :raises LocalVersionNotFound: If the specified version is not found locally.
        :raises LocalVersionDuplicated: If multiple versions matching the version parameter are found locally.
        """
        with self.lock:
            return self._get_version_manager(version).get_file(pattern)

    def list_versions(self) -> List[VersionManager]:
        """
        List all the local model versions managed by this ModelManager.

        :return: A list of VersionManager objects, one for each version.
        :rtype: List[VersionManager]
        """
        with self.lock:
            retval = []
            for version_name, version_id, version_dir in self._list_local_versions():
                retval.append(VersionManager(version_dir, self.model_name_or_id, self.creator,
                                             version_name, offline=True))

            return retval

    @property
    def total_size(self) -> int:
        """
        Get the total size of all the local model files managed by this ModelManager.

        :return: The total size in bytes.
        :rtype: int
        """
        return sum((version.total_size for version in self.list_versions()))

    def delete_version(self, version: Union[str, int, None] = None):
        """
        Delete the specified model version from the local storage.

        :param version: The version ID or name to delete. If None, the latest version is deleted.
        :type version: Union[str, int, None]
        :raises LocalVersionNotFound: If the specified version is not found locally.
        """
        with self.lock:
            version_name, version_id, version_dir = self._find_local_version(version)
            shutil.rmtree(version_dir, ignore_errors=True)

    def _tree(self):
        return self, [item._tree() for item in sorted(self.list_versions(), key=repr)]

    def _repr(self):
        return f'<{self.__class__.__name__} model: {self.model_name_or_id!r}>'

    def __str__(self):
        return format_tree(
            nested_map(repr, self._tree()),
            lambda x: x[0],
            lambda x: x[1],
        )

    def __repr__(self):
        return self._repr()
