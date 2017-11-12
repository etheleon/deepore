# Deepore: Deep learning for base calling MinION reads

The MinION device by Oxford Nanopore Technologies (ONT) is the first portable
USB sequencing device which promises play a unique part in the future of DNA sequencing.

Not only is it portable, the underlying technology is able to produce long reads (1Mb)
as compared to the current status quo of short reads (100 ~ 300 bp).

However it suffers from a high sequencing error rate.

The objective of this project is to apply deep neural network models to improve upon the base calling procedure.
Initial models were based on Hidden Markov Models (HMMs)
however several deep neural network implementations have already been published;
DeepNano (RNN) (Boža _et al_ 2017), Chiron (CNN + RNN) (Teng _et al_ 2017).

The problem of base calling in computational biology runs parallel to
machine translation in natural language processing (NLP) as both fields
 attempt to translate one sequence to another sequence.

Hence we can try to use cross-pollinate methods from both sides and see the results from this experiment.

```bash
git clone --recursive https://github.com/etheleon/deepore.git
```

## Docker Container

We are using nvidia's customised [docker](https://github.com/NVIDIA/nvidia-docker) `nvidia-docker`.
![nvidia-docker](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

Which is based on the `8.0-cudnn6-runtime-ubuntu16.04` tag

tensorflow version 1.3.0
python=2.7

We modified the docker from `https://github.com/anurag/fastai-course-1.git`


To start the container:

```
nvidia-docker run -it \
    --entrypoint /bin/zsh \
    -v /data/nanopore/new/fast5Dir/:/data \
    --name nanopore \
    -w /home/docker \
    -p 8889:8888 \
    etheleon/chiron
```

To start a new shell with a existing container running

```
containername="awesome_benz"
nvidia-docker exec -it $containername /bin/zsh
```

To train (ecoli), the model

1. run preprocessing first
2. run `chiron_rcnn_train.py`

but remember to check 2 things

1. set the the raw file directory, containing the `.signal` and `.label` files
2. the logs directory, by default this will be pointing to `/home/docker/out/logs`.
Remember to backup the contents of this folder if you're running a new model,
else the checkpoint data will saved over.

For the ecoli dataset, the raw files are in `/home/docker/ecoli/data/ecoli_raw`

```
➜  deepore git:(master) ✗ ls ~/ecoli/data/ecoli_raw | head
nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_40525_ch100_read381_strand1.label
nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_40525_ch100_read381_strand1.signal
nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_40525_ch100_read423_strand.label
nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_40525_ch100_read423_strand.signal
```

```
export CUDA_VISIBLE_DEVICES="1"
newChiron=</path/2/new/chiron>
python $newChiron/chiron/chiron_rcnn_train.py
```


To run original chiron the `8.0-cudnn5-runtime-ubuntu16.04` tag should be used since tensorflow 1.0.1 relies on cudnn5.

## Training data: Ecoli

Reference sequence NC_000913.fna
```
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/archive/old_refseq/Bacteria/Escherichia_coli_K_12_substr__MG1655_uid57779/NC_000913.fna
```

1. A subset of 254 reads from human genome (chromosome 12 part 9, chiron used chromosome 23 part 3) from the [nanopore WGS consortium](https://github.com/nanopore-wgs-consortium/NA12878) [need citation]
2. Ecoli reads in fast5 format from Nic Loman's lab [link](http://lab.loman.net/2016/07/30/nanopore-r9-data-release/) [need citation]



## Validation data: Zika




### Preprocessing

#### 1. Resquiggling

Based on proprietary basecalled sequence, we align using reference sequence NC_000913 to correct for basecall errors.

![alt text](https://github.com/etheleon/deepore/blob/master/misc/photo_2017-10-26_16-40-05.jpg)

```
bash ./preprocessing/resquiggle.sh
```

Rmbr to edit the variables in `resquiggle.sh`

| # reads | Failed Alignment |
| ---     | ---              |
| 164472  | 171              |

#### 2. Extracting the raw signal

```
bash ./preprocessing/runraw.sh
```



# Reference

Boža, V, Brejová, B, Vinař, T (2017). DeepNano: Deep recurrent neural networks for base calling in MinION nanopore reads. PLoS ONE, 12, 6:e0178751.

Teng, H, Hall, M B, Duarte, T, Cao, M D, Coin, L (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. bioRxiv, 
0
