for model in rnn birnn lstm bilstm
do
    for dim in 50 100 150 200
    do
        for t in 1 2 3 4 5
        do
            th run.lua restaurant-2015 $model $dim $t
        done
    done
done
