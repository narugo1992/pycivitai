import logging
import os
from functools import partial

import click
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word

from .dispatch import _get_global_manager
from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

try:
    import InquirerPy
except (ModuleNotFoundError, ImportError):
    InquirerPy = None

print_version = partial(_origin_print_version, 'pycivitai')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Command-Line Interface for pycivitai')
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


def _get_choices(choices):  # pragma: no cover
    from InquirerPy import inquirer
    return inquirer.checkbox(
        message='Choose model versions to delete:',
        choices=choices,
        cycle=False,
        transformer=lambda result: "%s region%s selected"
                                   % (len(result), "s" if len(result) > 1 else ""),
    ).execute()


def _confirm(text):  # pragma: no cover
    from InquirerPy import inquirer
    return inquirer.confirm(text).execute()


@cli.command('delete-cache', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Delete downloaded models from storage.')
@click.option('-A', '--all', 'delete_all', is_flag=True, type=bool, default=False,
              help='Delete all the downloaded models.', show_default=True)
def delete_cache(delete_all):
    manager = _get_global_manager(offline=True)
    with manager.lock:
        if delete_all:
            total_size = 0
            for model in manager.list_models():
                model_size = model.total_size
                logging.debug(f'Deleting {model.model_name_or_id!r}'
                              f'({size_to_bytes_str(model_size, precision=3)}) ...')
                manager.delete_model(model.model_name_or_id)
                total_size += model_size

            click.echo(f'All models deleted, total {size_to_bytes_str(total_size, precision=3)}.')

        else:
            if InquirerPy is None:
                raise OSError(
                    'TUI for models deletion not available, '
                    'please install CLI extra with `pip install pycivitiai[cli]`.'
                )

            from InquirerPy import inquirer
            from InquirerPy.base import Choice
            from InquirerPy.separator import Separator

            choices = []
            for model in manager.list_models():
                model_dir = model.root_dir
                model_name, model_id = os.path.basename(model_dir).split('__')

                model_total_size, model_files = 0, 0
                model_choices = []
                for version in model.list_versions():
                    version_dir = version.root_dir
                    version_name, version_id = os.path.basename(version_dir).split('__')
                    files = version.list_files()
                    total_size = sum([file.size for file in files])

                    if len(files) > 0:
                        model_files += len(files)
                        model_total_size += total_size
                        model_choices.append(Choice(
                            (model.model_name_or_id, version.version, len(files), total_size),
                            name=f'{version_name}(ID: {version_id}, {plural_word(len(files), "file")}, '
                                 f'size: {size_to_bytes_str(total_size, precision=3)})',
                            enabled=False
                        ))

                if model_choices:
                    choices.extend([
                        Separator(f'Model {model_name}(ID: {model_id}, {plural_word(model_files, "file")}, '
                                  f'size: {size_to_bytes_str(model_total_size, precision=3)}):'),
                        *model_choices,
                        Separator('')
                    ])

            if choices:
                versions_to_detect = _get_choices(choices)
                files_to_delete = sum((x[2] for x in versions_to_detect))
                size_to_delete = sum((x[3] for x in versions_to_detect))
                if files_to_delete == 0:
                    click.echo(click.style('Deletion cancelled.', fg='yellow'))
                elif _confirm(f'{plural_word(len(versions_to_detect), "version")} with '
                              f'{plural_word(files_to_delete, "file")} will be deleted, '
                              f'{size_to_bytes_str(size_to_delete, precision=3)} of disk usage '
                              f'will be released, confirm?'):
                    for model_name, version_name, _, _ in versions_to_detect:
                        manager.delete_version(model_name, version_name)
                    click.echo(click.style('Deletion complete!', fg='green'))

            else:
                click.echo(click.style('No models found to delete.', fg='yellow'))
