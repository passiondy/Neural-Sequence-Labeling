for t in {1..5}
do
    for d in 50 100 150
    do
        th run.lua restaurant-2015 crf $d $t
    done
done
