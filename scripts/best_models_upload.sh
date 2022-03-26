indir=$1

indir_tmp=${indir//_/-}
id=${indir_tmp#*/workspace/}


echo "{
  \"title\": \"${id}\",
  \"id\": \"jeamee/${id}\",
  \"licenses\": [
    {
      \"name\": \"CC0-1.0\"
    }
  ]
}" > ${indir}/dataset-metadata.json

kaggle datasets create -p $indir