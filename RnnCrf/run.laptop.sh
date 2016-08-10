for t in {1..5}
do
    for d in 50 100 150
    do
        th run.lua laptop crf $d $t
    done
done
