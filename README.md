# pycivitai

[![PyPI](https://img.shields.io/pypi/v/pycivitai)](https://pypi.org/project/pycivitai/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycivitai)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/83ada1e69b66b8f94b0440f27ced4548/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/83ada1e69b66b8f94b0440f27ced4548/raw/comments.json)

[![Code Test](https://github.com/narugo1992/pycivitai/workflows/Code%20Test/badge.svg)](https://github.com/narugo1992/pycivitai/actions?query=workflow%3A%22Code+Test%22)
[![Data Publish](https://github.com/narugo1992/pycivitai/actions/workflows/data.yml/badge.svg)](https://github.com/narugo1992/pycivitai/actions/workflows/data.yml)
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

Python Client and Model Management for civitai

Here is an example:

```python
from pycivitai.dispatch import civitai_download

if __name__ == '__main__':
    # get the latest version of DEN_barbucci_artstyle (either model title or id is okay)
    print(civitai_download('DEN_barbucci_artstyle'))

    # get the specific version (either version name of id is okay)
    print(civitai_download('DEN_barbucci_artstyle', version='v1.0'))

    # get the primary file of this model (it contains a ckpt and a vae, the ckpt is primary)
    print(civitai_download('Cetus-Mix'))  # the ckpt file
    print(civitai_download('Cetus-Mix', file='*.vae.pt'))  # get the vae file

```

If you need to delete the local models, just

```shell
pycivitai delete-cache -A  # download all models
```

or use the TUI to choose which one to delete

```shell
pip install pycivitai[cli]  # this step is necessary
pycivitai delete-cache
```

