# pycivitai

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
