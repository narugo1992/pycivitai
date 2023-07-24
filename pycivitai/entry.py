from functools import partial

import click

from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'pycivitai')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Command-Line Interface for pycivitai')
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover
