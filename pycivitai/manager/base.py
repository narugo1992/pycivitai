import re


def _soft_name_strip(name: str) -> str:
    return re.sub(r'[\W_]+', '_', name.lower()).strip('_')
