import fnmatch
import re
from dataclasses import dataclass
from typing import Union, Optional

from .http import get_session, ENDPOINT


class ModelNotFound(Exception):
    pass


class ModelFoundDuplicated(Exception):
    pass


class ModelVersionNotFound(Exception):
    pass


class ModelVersionDuplicated(Exception):
    pass


class ResourceNotFound(Exception):
    pass


class ResourceDuplicated(Exception):
    pass


@dataclass
class Resource:
    """
    Data class for resource.
    """
    model_name: str
    model_id: int
    creator: str
    version_name: str
    version_id: int
    filename: str
    is_primary: bool
    url: str
    sha256: str
    crc32: str
    hashes: dict
    size: int


def find_model_by_id(model_id) -> dict:
    """
    Retrieve model information from the CiviTAI API based on the given model ID.

    :param model_id: The ID of the model to retrieve information for.
    :type model_id: int
    :return: The dictionary containing the model information.
    :rtype: dict
    :raises ModelNotFound: If the model with the given ID is not found.
    """
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


def find_model_by_name(model_name: str, creator: Optional[str] = None) -> dict:
    """
    Retrieve model information from the CiviTAI API based on the given model name.

    :param model_name: The name of the model to retrieve information for.
    :type model_name: str
    :param creator: Name of creator. ``None`` means anyone.
    :type creator: Optional[str]
    :return: The dictionary containing the model information.
    :rtype: dict
    :raises ModelNotFound: If the model with the given name is not found.
    :raises ModelFoundDuplicated: If multiple models with the same name are found.
    """
    resp = get_session().get(f'{ENDPOINT}/api/v1/models', params={'query': model_name})
    resp.raise_for_status()
    collected_items = []
    for item in resp.json()['items']:
        if _name_strip(item['name']) == _name_strip(model_name) and \
                (creator is None or _name_strip(item['creator']['username']) == _name_strip(creator)):
            collected_items.append(item)

    if not collected_items:
        raise ModelNotFound(model_name)
    elif len(collected_items) > 1:
        all_names = [item['name'] for item in collected_items]
        raise ModelFoundDuplicated(model_name, all_names)
    else:
        return collected_items[0]


def find_model(model_name_or_id: Union[int, str], creator: Optional[str] = None) -> dict:
    """
    Find model information from the CiviTAI API based on the given model name or ID.

    :param model_name_or_id: The name or ID of the model to retrieve information for.
    :type model_name_or_id: Union[int, str]
    :param creator: Name of creator. ``None`` means anyone.
    :type creator: Optional[str]
    :return: The dictionary containing the model information.
    :rtype: dict
    :raises TypeError: If the model name or ID is not a valid integer or string.
    """
    if isinstance(model_name_or_id, int):
        return find_model_by_id(model_name_or_id)
    elif isinstance(model_name_or_id, str):
        if creator is None:
            try:
                model_id = int(model_name_or_id)
                return find_model_by_id(model_id)
            except (ModelNotFound, TypeError, ValueError):
                pass
        return find_model_by_name(model_name_or_id, creator)
    else:
        raise TypeError(f'Unknown model name or id, it should be an integer or string - {model_name_or_id!r}.')


def find_version(model_data: dict, version: Union[int, str, None] = None):
    """
    Find the specified version from the model data.

    :param model_data: The model data containing version information.
    :type model_data: dict
    :param version: The version ID or name to find. If None, the first version will be chosen.
    :type version: Union[int, str, None]
    :return: The dictionary containing the selected version information.
    :rtype: dict
    :raises ModelVersionNotFound: If the specified version is not found.
    :raises ModelVersionDuplicated: If multiple versions with the same ID or name are found.
    """
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


def find_resource(model_data: dict, version_data: dict, pattern: str = None):
    """
    Find the specified resource from the model and version data.

    :param model_data: The model data containing version information.
    :type model_data: dict
    :param version_data: The version data containing resource information.
    :type version_data: dict
    :param pattern: The pattern to match the resource filename. If None, the primary resource will be chosen.
    :type pattern: str
    :return: The resource information.
    :rtype: Resource
    :raises TypeError: If the pattern of the resource is not a valid string.
    :raises ResourceNotFound: If the specified resource is not found.
    :raises ResourceDuplicated: If multiple resources with the same name are found.
    """
    # find chosen resource
    if pattern is None:
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
        creator=model_data['creator']['username'],
        version_name=version_data['name'], version_id=version_data['id'],
        filename=select_file['name'],
        url=select_file['downloadUrl'],
        sha256=select_file['hashes']['SHA256'],
        crc32=select_file['hashes']['CRC32'],
        hashes=select_file['hashes'],
        is_primary=select_file.get('primary', False),
        size=file_size,
    )
