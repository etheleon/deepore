#g1Dir=/data/nanopore/newfast5Dir2
#genomeFn=/data/nanopore/chr12.fna

#ecoli
#wget ftp://ftp.ncbi.nlm.nih.gov/genomes/archive/old_refseq/Bacteria/Escherichia_coli_K_12_substr__MG1655_uid57779/NC_000913.fna
#g1Dir=/data/nanopore/data/ecoli/downloads/pass
#genomeFn=/data/nanopore/data/NC_000913.fna

#zika
g1Dir=/data/datasets/zika
genomeFn=/home/uesu/githubb/deepore/referenceGenome/nc_012532.1.fasta


nanoraw genome_resquiggle \
      $g1Dir $genomeFn --graphmap-executable /home/uesu/local/graphmap/bin/graphmap \
      --timeout 60 --cpts-limit 100 --normalization-type median \
      --failed-reads-filename /data/nanopore/nanologs/zika.testing.signif_group1.failed_read.txt \
      --processes 20 --overwrite
