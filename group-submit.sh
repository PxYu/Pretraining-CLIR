model_type="mbert"
dataset="mix"
batch_size=64

for source_lang in "en" "es" "fr" "de"
do
    for target_lang in "en" "es" "fr" "de"
    do
        if [ $source_lang != $target_lang ]; then
	    sbatch group.sh $model_type $dataset $source_lang $target_lang $batch_size
        fi
    done
done
