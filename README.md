# pycivitai

[![PyPI](https://img.shields.io/pypi/v/pycivitai)](https://pypi.org/project/pycivitai/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycivitai)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/83ada1e69b66b8f94b0440f27ced4548/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/83ada1e69b66b8f94b0440f27ced4548/raw/comments.json)

[![Code Test](https://github.com/narugo1992/pycivitai/workflows/Code%20Test/badge.svg)](https://github.com/narugo1992/pycivitai/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/narugo1992/pycivitai/workflows/Package%20Release/badge.svg)](https://github.com/narugo1992/pycivitai/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/narugo1992/pycivitai/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/narugo1992/pycivitai)

![GitHub Org's stars](https://img.shields.io/github/stars/narugo1992)
[![GitHub stars](https://img.shields.io/github/stars/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/narugo1992/pycivitai)
[![GitHub issues](https://img.shields.io/github/issues/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/pulls)
[![Contributors](https://img.shields.io/github/contributors/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/narugo1992/pycivitai)](https://github.com/narugo1992/pycivitai/blob/master/LICENSE)

Python Client and Model Management for civitai,

The design was inspired by [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub), implementing
automatic management of downloaded models locally, and automatically detecting file changes and the latest versions.

## Installation

You can install `pycivitai` with pip

```shell
pip install pycivitai
```

If you need TUI for manually deleting downloaded models, just install like this

```shell
pip install pycivitai[cli]
```

## Quick Start

### Use Model From CivitAI

`pycivitai` follows a similar usage pattern as the `huggingface_hub`, as shown below:

```python
from pycivitai import civitai_download

if __name__ == '__main__':
    # get the latest version of DEN_barbucci_artstyle (either model title or id is okay)
    # Homepage of this model: https://civitai.com/models/85716
    # the return value of civitai_download is the path of downloaded file you need
    print(civitai_download('DEN_barbucci_artstyle'))

    # get the same model (DEN_barbucci_artstyle) with model id
    print(civitai_download(85716))

    # get the specific version (either version name of id is okay)
    print(civitai_download('DEN_barbucci_artstyle', version='v1.0'))
    print(civitai_download('DEN_barbucci_artstyle', 91158))

    # This is a model with multiple files inside.
    # It contains a safetensors and a vae, the safetensors is primary.
    # Homepage of this model: https://civitai.com/models/6755/cetus-mix
    print(civitai_download('Cetus-Mix'))  # the safetensors file (when file not specified, primary file will be chosen)
    print(civitai_download('Cetus-Mix', file='*.vae.pt'))  # get the vae file

```

If you are using command line, you can get the models with the following commands

```shell
pycivitai get -m 'DEN_barbucci_artstyle'           # get model, use primary file of the latest version
pycivitai get -m 'DEN_barbucci_artstyle' -v 'v1.0' # get model, use primary file the v1.0 version
pycivitai get -m 'Cetus-Mix'                       # get the safetensors file of Cetus-Mix
pycivitai get -m 'Cetus-Mix' -f '*.vae.pt'         # get the vae file of Cetus-Mix

```

### Get Information of Model Resource

If you only need to obtain information about the model's resource files, for example, if you need to download the files
yourself, you can use the `civitai_find_online` function, which will return a `Resource` object.

```python
from pycivitai import civitai_find_online, Resource

if __name__ == '__main__':
    # get resource of this model file, arguments are the same as `civitai_download`
    resource: Resource = civitai_find_online('DEN_barbucci_artstyle')

    # information of this resource
    print(resource.model_name)
    print(resource.model_id)
    print(resource.version_name)
    print(resource.version_id)
    print(resource.filename)
    print(resource.url)
    print(resource.size)
    print(resource.sha256)
    print(resource.is_primary)

```

### Clear the Downloaded Models

If you need to delete all the local models, just

```shell
pycivitai delete-cache -A  # download all models
```

or use the TUI to choose which one to delete

```shell
pip install pycivitai[cli]  # this step is necessary
pycivitai delete-cache
```

TUI is like this, build with [InquirerPy](https://github.com/kazhala/InquirerPy):

```
? Choose model versions to delete:
  Model cetus_mix(ID: 6755, 2 files, size: 3.899 GiB):
❯ ○ cetusmix_whalefall2(ID: 105924, 2 files, size: 3.899 GiB)

  Model den_barbucci_artstyle(ID: 85716, 2 files, size: 17.181 KiB):
  ○ v1_0(ID: 91158, 1 file, size: 7.160 KiB)
  ○ v2_0(ID: 113049, 1 file, size: 10.021 KiB)
```

## F.A.Q.

### Where will the downloaded model be saved?

The downloaded model will be saved by default in the `~/.cache/civitai` directory. If you need to change the save path,
you can set the value of the `CIVITAI_HOME` environment variable to your desired save path.

### How are downloaded models managed?

After downloading, the models are managed using the file system, and file locks are set to ensure thread safety during
reading and writing. The structure is similar to the following:

```
~/.cache/civitai
├── cetus_mix__6755
│   └── cetusmix_whalefall2__105924
│       ├── files
│       │   ├── cetusMix_Whalefall2.safetensors
│       │   └── vae-ft-mse-840000-ema-pruned.vae.pt
│       ├── hashes
│       │   ├── cetusMix_Whalefall2.safetensors.hash
│       │   └── vae-ft-mse-840000-ema-pruned.vae.pt.hash
│       └── primary
└── den_barbucci_artstyle__85716
    ├── v1_0__91158
    │   ├── files
    │   │   └── DEN_barbucci_artstyle.pt
    │   ├── hashes
    │   │   └── DEN_barbucci_artstyle.pt.hash
    │   └── primary
    └── v2_0__113049
        ├── files
        │   └── DEN_barbucci_styleMK2.pt
        ├── hashes
        │   └── DEN_barbucci_styleMK2.pt.hash
        └── primary
```

This structure does not have any additional dependencies. Therefore, when it is necessary to migrate the storage path,
you can simply move it and modify the environment variable `CIVITAI_HOME`.

