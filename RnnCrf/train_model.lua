train_model = {}
function train_model.train(data, config)
    local tag_train = data.tag_train
    local seq_train = data.seq_train
    local valid
    if data.tag_dev and data.seq_dev then
        valid = {}
        valid.tag = data.tag_dev
        valid.seq = data.seq_dev
    end
    local test
    if data.tag_test and data.seq_test then
        test = {}
        test.tag = data.tag_test
        test.seq = data.seq_test
    end

    local N = #seq_train
    local maxIter = config.maxIter
    local bSize = config.bSize
    local bNum = torch.ceil(N/bSize)
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    local best_model
    local best_perf = 0
    local best_perf_test = 0
    local perf_valid
    local perf_record
    local perf_record_test
    local model = build_model(config)
    model:training()
    local p,dp = model:parameters()
    print(p)
    print(dp)
    print(model)
    local para, dpara = model:getParameters()
    print(para:size())
    print(dpara:size())
    local lr = 0.01

    local optim_config = config.optim_config
    for iter=1, maxIter do
        local idxList = torch.randperm(N)
        local allLoss = 0
        local start_time = sys.clock()
        print('----------- Iter  '..iter..' ---------')
        for b = 1,bNum do
            local numWord = 0
            xlua.progress(b, bNum)
            model:zeroGradParameters()
            local feval = function()
                local bLoss = 0
                for i=(b-1)*bSize+1, math.min(b*bSize, N) do
                    local idx = idxList[i]
                    local z = model:predict(seq_train[idx])
                    local loss = -model:forward(seq_train[idx], z)
                    model:backward(seq_train[idx], z, -1)
                    loss = loss + model:forward(seq_train[idx], tag_train[idx])
                    model:backward(seq_train[idx], tag_train[idx], 1)
                    loss = loss + torch.ne(torch.Tensor(z), torch.Tensor(tag_train[idx])):sum()
                    bLoss = bLoss + loss
                    numWord = numWord + #seq_train[idx]
                end
                allLoss = allLoss+bLoss
            return bLoss, dpara
            end
            optim.adagrad(feval, para, optim_config)
            --feval()
            --model:updateParameters(lr)
        end
        print('Elapsed time: ', sys.clock()-start_time)
        print('Obj func value:', allLoss)
        print('Sum square of p:', torch.pow(para,2):sum()*optim_config.weightDecay)
        local perf
        local perf_test
        local perf_train = train_model.cal_performance(model, {tag=data.tag_train, seq=data.seq_train})
        print('Train f-score:', perf_train[3])
        print(perf_train[1], perf_train[2])
        print(perf_train[6], perf_train[7])
        if valid then
            perf = train_model.cal_performance(model, valid)
            print('Valid f-score:', perf[3])
            print(perf[1], perf[2])
            print(perf[6], perf[7])
        end
        if test then
            perf_test = train_model.cal_performance(model, test)
            print('Test f-score:', perf_test[3])
            print(perf_test[1], perf_test[2])
            print(perf_test[6], perf_test[7])
        end
        if valid and perf[3] > best_perf then
            best_model = model:clone()
            best_perf = perf[3]
            print('Get the best performance until now: ', best_perf)
            perf_valid = perf
            perf_record = perf_test
        end
        if perf_test[3] > best_perf_test then
            best_perf_test = perf_test[3]
            perf_record_test = perf_test
        end
    end
    --torch.save('best-model', best_model)
    return best_model, perf_valid, perf_record, perf_record_test
end

function train_model.cal_performance(model, data)
    local seq = data.seq
    local tag = data.tag
    local try, truth, get, try_t, truth_t, get_t, shoot, miss = 0,0,0,0,0,0,0,0
    local predictions = {}
    for i=1,#seq do
        local pred_tags = model:predict(seq[i])
        --print(pred_tags)
        --local pred_tags = evaluate_func.get_prediction_tags(output)
        table.insert(predictions, pred_tags)
        local result = evaluate_func.evaluate_list(pred_tags, tag[i])
        try = try+result[1]
        truth = truth+result[2]
        get = get+result[3]
        try_t = try_t+result[4]
        truth_t = truth_t+result[5]
        get_t = get_t+result[6]
        shoot = shoot+result[7]
        miss = miss + result[8]
    end
    --print(try, truth, get)
    return {get/try, get/truth, 2*get/(try+truth), get_t/try_t, get_t/truth_t, get_t*2/(truth_t+try_t), shoot/(shoot+miss)}, predictions
end
