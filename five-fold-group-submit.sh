model_type="mbert"
dataset="clef"
batch_size=8

for source_lang in "en" "es" "fr" "de"
do
    for target_lang in "en" "es" "fr" "de"
    do
        if [ $source_lang != $target_lang ]; then
	    sbatch five-fold-group.sh $model_type $dataset $source_lang $target_lang $batch_size
        fi
    done
done