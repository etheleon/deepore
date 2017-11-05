#human
#fast5=/data/nanopore/newfast5Dir2/
#output=/data/nanopore/out/rawoutput
#ecoli

fast5=/data/nanopore/data/ecoli/downloads/pass
output=/data/nanopore/data/ecoli_raw
chironDir=$HOME/github

python $chironDir/chiron/utils/raw.py \
    --input $fast5 \
    --output $output \
    --basecall_group RawGenomeCorrected_000
