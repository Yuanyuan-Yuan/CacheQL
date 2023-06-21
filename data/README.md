# Data

## Secret Data

You need to first create the following directories in this folder.

- `celeba_crop128` - The processed CelebA dataset.

- `key/libgcrypt-rsa` - The private keys of Libgcrypt RSA.

- `key/openssl-rsa` - The private keys of OpenSSL/MbedTLS RSA.

- `key/openssl-aes` - The private keys of OpenSSL/MbedTLS AES.

- `npz/rsa` - The converted keys (in `.npz` format) of RSA.

- `npz/aes` - The converted keys (in `.npz` format) of AES.

## Side Channel Traces 

For side channel traces, you need to create the following directories.

- `trace/SETTING/SOFTWARE/cachebank` - The collected side channel records of the whole execution for different software under different settings (blinding enabled).

- `trace/SETTING/{SOFTWARE or libjpeg-turbo-2.1.2}/cacheline` - The collected side channel records of the whole execution for different software under different settings (blinding enabled).

- `trace/pp_dcache/{SOFTWARE or libjpeg-turbo-2.1.2}` - The collected side channel records of the whole execution for different software under different settings (blinding enabled).

- `trace/pp_icache/{SOFTWARE or libjpeg-turbo-2.1.2}` - The collected side channel records of the whole execution for different software under different settings (blinding enabled).

- `trace_det/SETTING/SOFTWARE/cacheline` - The collected side channel records of the whole execution with blinding disabled.

- `trace_dec/SETTING/SOFTWARE/cacheline` - The collected side channel records of the decryption phase (blinding enabled).

- `trace_dec_det/SETTING/SOFTWARE/cacheline` - The collected side channel records of the decryption phase with blinding disabled.

Choose `SETTING` from [`SDA`, `SCB`].

Choose `SOFTWARE` from the following:

```python
[
    'rsa_openssl_0.9.7c', 'rsa_openssl_3.0.0',
    'rsa_mbedtls_2.15.0', 'rsa_mbedtls_3.0.0',
    'rsa_sign_libgcrypt_1.6.1', 'rsa_sign_libgcrypt_1.9.4',
    'aes_openssl_0.9.7c', 'aes_openssl_3.0.0',
    'aes_mbedtls_2.15.0', 'aes_mbedtls_3.0.0',
]
```