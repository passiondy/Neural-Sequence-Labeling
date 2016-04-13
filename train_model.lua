train_model = {}
function train_model.train(data, config)
    local tag_train = data.tag_train
	local seq_train = data.seq_train
    local valid = {}
    valid.tag = data.tag_dev
	valid.seq = data.seq_dev
	local training = {}
	training.seq = seq_train
	training.tag = tag_train
    local N = #seq_train
    local maxIter = config.maxIter
    local bSize = config.bSize
    local bNum = torch.ceil(N/bSize)
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    local best_model
    local best_perf = 0
    local model = build_model(config)
    local para, dpara = model:getParameters()
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
                    local output = model:forward(seq_train[idx])
                    local loss = criterion:forward(output, tag_train[idx])
                    local gradOut = criterion:backward(output, tag_train[idx])
                    model:backward(seq_train[idx], gradOut)
                    bLoss = bLoss + loss
					numWord = numWord + #seq_train[idx]
                end
                allLoss = allLoss+bLoss
			return bLoss, dpara
            end
            optim.adagrad(feval, para, optim_config)
			--model:updateParameters(lr/numWord)
        end
        print('Elapsed time: ', sys.clock()-start_time)
        local perf = train_model.cal_performance(model, valid)
		print(allLoss)
		print(perf[3])
        if perf[3] > best_perf then
            best_model = model:clone()
            best_perf = perf[3]
			print('Get the best performance until now: ', best_perf)
		--[[else
			break]]
        end
    end
    return best_model
end

function train_model.cal_performance(model, data)
    local seq = data.seq
    local tag = data.tag
    local try, truth, get, try_t, truth_t, get_t, shoot, miss = 0,0,0,0,0,0,0,0
    for i=1,#seq do
        local output = model:forward(seq[i])
        local pred_tags = evaluate_func.get_prediction_tags(output)
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
	print(try, truth, get)
    return {get/try, get/truth, 2*get/(try+truth), get_t/try_t, get_t/truth_t, get_t*2/(truth_t+try_t), shoot/(shoot+miss)}
end
