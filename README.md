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

## Docker

Use `nvidia-docker`

we modified the docker from `https://github.com/anurag/fastai-course-1.git`


```
nvidia-docker run -it \
    --entrypoint /bin/zsh \
    -v /data/nanopore/new/fast5Dir/:/data \
    --name nanopore \
    -w /home/docker \
    -p 8889:8888 \
    -w /home/docker \
    etheleon/chiron
```

To train deepore we need to run chiron_rcnn_train.py

```
python Chiron/chiron/chiron_rcnn_train.py
```


## Training data: Ecoli

### Reference sequence NC_000913.fna

```
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/archive/old_refseq/Bacteria/Escherichia_coli_K_12_substr__MG1655_uid57779/NC_000913.fna
```

1. A subset of 254 reads from human genome (chromosome 12 part 9, chiron used chromosome 23 part 3) from the [nanopore WGS consortium](https://github.com/nanopore-wgs-consortium/NA12878) [need citation]
2. Ecoli reads in fast5 format from Nic Loman's lab [link](http://lab.loman.net/2016/07/30/nanopore-r9-data-release/) [need citation]

### Preprocessing

#### 1. Resquiggling

Based on proprietary basecalled sequence, we align using reference sequence NC_000913 to correct for basecall errors.

![alt text](https://github.com/etheleon/deepore/blob/master/misc/photo_2017-10-26_16-40-05.jpg)

```
bash ./preprocessing/resquiggle.sh
Getting file list.
Correcting 164472 files with 1 subgroup(s)/read(s) each (Will print a dot for each 100 files completed).
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
....................................................................................................
......................................
Failed reads summary:
        Reached maximum number of changepoints for a single indel :     11309
                Alignment not produced. Potentially failed to locate BWA index files. : 171
```

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
