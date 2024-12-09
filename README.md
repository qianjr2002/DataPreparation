# DataPreparation

原来的目录树 结构
```
$ tree -d VCTK-DEMAND/
VCTK-DEMAND/               # 16kHz resampled
|-- test 
|   |-- clean              # clean_testset_wav.zip (147.1Mb)
|   `-- noisy              # noisy_testset_wav.zip (162.6Mb)
`-- train
    |-- clean              # clean_trainset_28spk_wav.zip (2.315Gb)
    `-- noisy              # noisy_trainset_28spk_wav.zip (2.635Gb)
```

考虑将这些`.wav`文件制作成 1份`VCTK_28spk.hdf5`文件 和 1份`VCTK_28spk.csv `索引文件，加快读取文件的速度
