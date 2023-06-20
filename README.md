
# CacheQL
Research Artifact of USENIX Security 2023 Paper: *CacheQL: Quantifying and Localizing Cache Side-Channel Vulnerabilities in Production Software*

Preprint: https://arxiv.org/pdf/2209.14952.pdf

## Note

**Warning**: This repo is provided as-is and is only for research purposes. Please use it only on test systems with no sensitive data. You are responsible for protecting yourself, your data, and others from potential risks caused by this repo.


## Installation

- Build from source code

    ```setup
    git clone https://github.com/Yuanyuan-Yuan/CacheQL
    cd CacheQL
    pip install -r requirements.txt
    ```

## Struture

This repo is organized as follows:

- `data` - This folder provides scripts for processing data.

- `pin` - This folder provides our pintools for logging secret-dependent data access (SDA) and control branch (SCB).

- `pp` - This folder provides scripts for logging cache set accesses with Prime+Probe.

- `software` - Our evaluated software.

- `config.py` - This script sets all global paths for different experiments.

- `params.py` - This script includes all parameters in our experiments.

- `model.py` - This script includes implementations of our encoder, compressor, and classifier.

- `dataset.py` - This script implements Pytorch dataset classes for different side channel traces and secrets.

- `utils.py` - Some utility tools.

- `MI*.py` - The script for quantifying side channel leaks under different scenarios.

- `shapley.py` - The script for localizing leakage sites in software.

## Software

Our evaluated software can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yyuanaq_connect_ust_hk/Emlm86l_hxJItXhs3OMItV4BbRdFGvwdF0unYqcVOKnyHA). Once downloaded, put everything into the `software` folder.

See [SOFTWARE](https://github.com/Yuanyuan-Yuan/CacheQL/tree/main/software) for detailed documents.

## Preparation

You need to first set up all related paths in `config.py`. We have set some default paths w.r.t. this project; you may need to update them if you want to analyze your own data and software.

See [DATA](https://github.com/Yuanyuan-Yuan/CacheQL/tree/main/data) for how the `data` folder is organized according to the current `config.py`.

## Datasets

Each dataset is implemented as one Pytorch Dataset class in `dataset.py`.

### Crypto Datasets

#### Preparation

```bash
cd data
python generate_key.py
```

Run the above commands to generate private keys for OpenSSL, MbedTLS, and Libgcrypt.
`generate_key.py` also converts the generated keys into `.npz` format for support of being loaded by Pytorch.

#### Implementations

`dataset.py` implements the following four datasets for crypto software. You can use them as Pytorch Datasets.

- `RSADatasetMulti` - For analyzing the whole trace of crypto software (logged by Pin).

- `RSADatasetMultiDec` - For the decryption phase (logged by Pin).

- `RSADatasetMultiPre` -For the pre-processing phase (logged by Pin).

- `RSADatasetMultiPP` - For traces collected using Prime+Probe.

### Media Datasets

#### Preparation

We use CelebA as one representative media dataset, which contains human face photos. The official dataset can be downloaded [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

After downloading the dataset, run the following commonds:

```bash
cd data
python celeba_process.py
```

The dataset will be splitted into different subfolders and be cropped and resized into $128 \times 128$. Several processed examples are given in the `data/celeba_crop128/fit` folder.

#### Implementation

The dataset class for CelebA is implemented using the `ImageDatasetMulti` class in `dataset.py`. You can use it as one Pytorch Dataset.

## Logging Side Channel Traces

### Intel Pin

We provide our pintools in the `pin` folder. 

- `SCB.cpp` - Logging records of secret-dependent control branch (SCB) for the whole execution.

- `SDA.cpp` - Logging records of secret-dependent data access (SDA) for the whole execution.

- `SCB_dec.cpp` - Logging records of SCB for the decryption phase.

- `SDA_dec.cpp` - Logging records of SDA for the decryption phase.

Download Pin from [here](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html) and unzip it to `PIN_ROOT` (specify this path by yourself). Then, run the following commands:

```bash
cp -r pin/*.cpp /PIN_ROOT/source/tools/ManualExamples/
cd /PIN_ROOT/source/tools/ManualExamples/
make obj-intel64/SCB.so TARGET=intel64
make obj-intel64/SDA.so TARGET=intel64
make obj-intel64/SCB_dec.so TARGET=intel64
make obj-intel64/SDA_dec.so TARGET=intel64
```

Before collecting side channel traces, run the following command to disable ASLR.

```bash
setarch $(uname -m) -R /bin/bash
```

When collecting traces for the decryption phase, you need to specify any one of the following arguments:

- `sr` - The name of the function starting decryption.

- `sa` - The address of the function starting decryption.

You can use the scripts provided [here](https://github.com/Yuanyuan-Yuan/Manifold-SCA/tree/main/pin) (following the instructions [here](https://github.com/Yuanyuan-Yuan/Manifold-SCA/tree/main#3-side-channel-attack)) to automate the collection of side channel traces.


### Prime+Probe

We use [Mastik](https://cs.adelaide.edu.au/~yval/Mastik/) (Ver. 0.02) to launch Prime+Probe on L1 cache to collect the cache set accesses. We provide our scripts in `pp/Mastik`. 
We recommend you setting the cache miss threshold in these scripts according to your machines.

Suppose Mastik is downloaded into the path `/MASTIK`. Run the following commands:

```bash
cp -r pp/Mastik/*.c /MASTIK/demo
cd /MASTIK
make
```

We assume victim and spy are on the same CPU core and no other process is runing on this CPU core. 

First, run the following command to isolate the `cpu_id`-th CPU core .

```bash
sudo cset shield --cpu {cpu_id}
```

Then collect side channel records as the follow.

```bash
cd Mastik
sudo cset shield --exec python run_pp.py -- {pp_crypto OR pp_media} {cpu_id} {segment_id}
```

`Mastik/pp_crypto.py`/`Mastik/pp_media.py` is the coordinator which runs spy and victim on the same CPU core simultaneously and saves the collected cache set access.

## Quantifying Leaks

### Crypto Software

Run `python MI.py --software XXX --side YYY --setting ZZZ` to train a model and quantify leaks for the whole trace.

- `--software` - The analyzed software. 
```python
choices = [
    'rsa_openssl_0.9.7c', 'rsa_openssl_3.0.0',
    'rsa_mbedtls_2.15.0', 'rsa_mbedtls_3.0.0',
    'rsa_sign_libgcrypt_1.6.1', 'rsa_sign_libgcrypt_1.9.4',
    'aes_openssl_0.9.7c', 'aes_openssl_3.0.0',
    'aes_mbedtls_2.15.0', 'aes_mbedtls_3.0.0',
]
```

- `--side` - The type of side channels logged via Pin.  
choices = [`cacheline`, `cachebank`]

- `--setting` - The setting of the leakage mode.  
choices = [`SDA`, `SCB`]

### Crypto Software w/o Blinding

Run `python MI_det.py --software XXX --side YYY --setting ZZZ` to quantify leaks for crypto software whose blinding is disabled. The choices of parameters are same as the above.

Since side channel traces are deterministic, validation is not needed.

### Pre-Processing/Decryption Phase

Run `python MI_dec.py --software XXX --side YYY --setting ZZZ` to quantify leaks for the decryption phase.

Run `python MI_dec_det.py --software XXX --side YYY --setting ZZZ` to quantify leaks for the decryption phase with blinding disabled.

The choices of parameters are same as the above.

### Media Software

Run `python MI_image.py --software libjpeg-turbo-2.1.2 --side YYY --setting ZZZ` to train a model and quantify leaks for the whole trace of `libjpeg`.

- `--side` - The type of side channels logged via Pin.  
choices = [`cacheline`, `cachebank`]

- `--setting` - The setting of leakage mode.  
choices = [`SDA`, `SCB`]

### Real Attack Logs

Run `python MI_image.py --software XXX --setting ZZZ --repeat_num N` to train a model and quantify leaks in side channel records logged via Prime+Probe.

- `--software` - The analyzed software. 
```python
choices = [
    'rsa_openssl_0.9.7c', 'rsa_openssl_3.0.0',
    'rsa_mbedtls_2.15.0', 'rsa_mbedtls_3.0.0',
    'rsa_sign_libgcrypt_1.6.1', 'rsa_sign_libgcrypt_1.9.4',
    'aes_openssl_0.9.7c', 'aes_openssl_3.0.0',
    'aes_mbedtls_2.15.0', 'aes_mbedtls_3.0.0',
]
```

- `--setting` - The setting of leakage.  
choices = [`pp_dcache`, `pp_icache`]

- `--repeat_num` - The number of repeats when performing Prime+Probe.  
choices = [`1`, `2`, `4`, `8`, `16`]

## Localizing Leakage Sites

Run `python shapley --software XXX --side YYY --setting ZZZ --use_IG 0` to localize the leakage sites.

- `--use_IG` - Whether using Integrated Gradient to compute the gradients.  
choices = [`0`, `1`]. Default is `0`, which uses the conventional method to compute gradients.

## Findings

The full report of our localized side channel vulnerabilities is provided at the [project page](https://sites.google.com/view/cache-ql#h.pgsarsaxsdsv).


## Acknowledgement

We sincerely thank Janos Follath (developer of MbedTLS), Matt Caswell (developer of OpenSSL), and developers of Libjpeg-turbo for their prompt responses and comments on our reported vulnerabilities.

## Citation

If CacheQL is helpful for your research, please consider cite our work as follows:

```bib
@inproceedings{yuan2023cacheql,
  title={CacheQL: Quantifying and Localizing Cache Side-Channel Vulnerabilities in Production Software},
  author={Yuan, Yuanyuan and Liu, Zhibo and Wang, Shuai},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  year={2023}
}
```

If you have any questions, feel free to contact Yuanyuan (yyuanaq@cse.ust.hk).