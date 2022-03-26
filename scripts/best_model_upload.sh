indir=$1
file=$2

outdir=${indir//_/-}
outfile=${file: 0 :11}

mkdir $outdir
ln ${indir}/${file} ${outdir}/${outfile}

id=${outdir#*/workspace/}-${file: 6 :1}

echo "{
  \"title\": \"${id}\",
  \"id\": \"jeamee/${id}\",
  \"licenses\": [
    {
      \"name\": \"CC0-1.0\"
    }
  ]
}" > ${outdir}/dataset-metadata.json

kaggle datasets create -p $outdir