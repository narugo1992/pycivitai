import os
from typing import Union, Optional, Iterator, Tuple, List

import requests
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
    def __init__(self, root_dir: str, model_name_or_id: Union[str, int],
                 model_data: Optional[dict] = None, offline: bool = False):
        self.root_dir = root_dir
        self.model_name_or_id = model_name_or_id
        self._model_data = model_data

        os.makedirs(root_dir, exist_ok=True)
        self._d_versions = os.path.join(self.root_dir)
        self._offline = offline

    def _get_model(self):
        if not self._model_data:
            self._model_data = find_model(self.model_name_or_id)
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
                version_dir, self.model_name_or_id, version_name,
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
            return VersionManager(version_dir, self.model_name_or_id, version_name, offline=True)

    def get_file(self, version: Union[str, int, None] = None, pattern: str = ...):
        return self._get_version_manager(version).get_file(pattern)

    def list_versions(self) -> List[VersionManager]:
        retval = []
        for version_name, version_id, version_dir in self._list_local_versions():
            retval.append(VersionManager(version_dir, self.model_name_or_id, version_name, offline=True))

        return retval

    def _tree(self):
        return self, [item._tree() for item in self.list_versions()]

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
