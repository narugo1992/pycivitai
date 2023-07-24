import fnmatch
import re
from dataclasses import dataclass
from typing import Union

from .http import get_session, ENDPOINT


class ModelNotFound(Exception):
    pass


class ModelFoundDuplicated(Exception):
    pass


def find_model_by_id(model_id) -> dict:
    resp = get_session().get(f'{ENDPOINT}/api/v1/models/{model_id}')
    if resp.ok:
        return resp.json()
    else:
        if resp.status_code == 404:
            raise ModelNotFound(model_id)
        else:
            resp.raise_for_status()


def _name_strip(name: str) -> str:
    return re.sub(r'[\W_]+', '', name.lower())


def find_model_by_name(model_name: str) -> dict:
    resp = get_session().get(f'{ENDPOINT}/api/v1/models', params={'query': model_name})
    resp.raise_for_status()
    collected_items = []
    for item in resp.json()['items']:
        if _name_strip(item['name']) == _name_strip(model_name):
            collected_items.append(item)

    if not collected_items:
        raise ModelNotFound(model_name)
    elif len(collected_items) > 1:
        all_names = [item['name'] for item in collected_items]
        raise ModelFoundDuplicated(model_name, all_names)
    else:
        return collected_items[0]


def find_model(model_name_or_id: Union[int, str]) -> dict:
    if isinstance(model_name_or_id, int):
        return find_model_by_id(model_name_or_id)
    elif isinstance(model_name_or_id, str):
        try:
            model_id = int(model_name_or_id)
            return find_model_by_id(model_id)
        except (ModelNotFound, TypeError, ValueError):
            return find_model_by_name(model_name_or_id)
    else:
        raise TypeError(f'Unknown model name or id, it should be an integer or string - {model_name_or_id!r}.')


@dataclass
class Resource:
    model_name: str
    model_id: int
    version_name: str
    version_id: int
    filename: str
    is_primary: bool
    url: str
    sha256: str
    size: int


class ModelVersionNotFound(Exception):
    pass


class ModelVersionDuplicated(Exception):
    pass


class ResourceNotFound(Exception):
    pass


class ResourceDuplicated(Exception):
    pass


def find_version(model_data: dict, version: Union[int, str, None] = None):
    # find chosen version
    versions = model_data['modelVersions']
    if version is None:
        select_version = versions[0]
    else:
        all_select_versions = []
        for vitem in versions:
            if vitem['id'] == version or _name_strip(vitem['name']) == _name_strip(str(version)):
                all_select_versions.append(vitem)

        if not all_select_versions:
            raise ModelVersionNotFound(model_data['name'], version)
        elif len(all_select_versions) > 1:
            raise ModelVersionDuplicated(model_data['name'], [vitem['name'] for vitem in all_select_versions])
        else:
            select_version = all_select_versions[0]

    return select_version


def find_resource(model_data: dict, version_data: dict, pattern: str = ...):
    # find chosen resource
    if pattern is ...:
        pattern, primary = '*', True
    elif isinstance(pattern, str):
        primary = False
    else:
        raise TypeError(f'Pattern of resource should be a string, but {pattern!r} found.')

    all_select_files = []
    for file in version_data['files']:
        if (primary and file.get('primary')) or (not primary and fnmatch.fnmatch(file['name'], pattern)):
            all_select_files.append(file)
    if not all_select_files:
        raise ResourceNotFound(model_data['name'], version_data['name'], {'pattern': pattern, 'primary': primary})
    elif len(all_select_files) > 1:
        raise ResourceDuplicated(model_data['name'], version_data['name'], [file['name'] for file in all_select_files])
    else:
        select_file = all_select_files[0]

    file_size = select_file['sizeKB'] * 1024
    assert abs(round(file_size) - file_size) < 1e-4
    file_size = int(round(file_size))
    return Resource(
        model_name=model_data['name'], model_id=model_data['id'],
        version_name=version_data['name'], version_id=version_data['id'],
        filename=select_file['name'],
        url=select_file['downloadUrl'],
        sha256=select_file['hashes']['SHA256'],
        is_primary=select_file.get('primary', False),
        size=file_size,
    )
