experiment = {}

function experiment.save_result(perf_list, fname)
    print(perf_list)
    local lines = {}
    for i,_ in pairs(perf_list) do
        local perf = perf_list[i]
        for v,p in pairs(perf) do
            table.insert(lines, p)
        end
        if i < #perf_list then
            table.insert(lines, '')
        end
    end
    utils.writefile(fname, stringx.join('\n',lines))
end

function experiment.save_predictions(predictions, tag_dict, fname)
    local lines = {}
    local rev_tag_dict = {}
    for v,p in pairs(tag_dict) do
        rev_tag_dict[p] = v
    end
    for _,pred in pairs(predictions) do
        local pred_str = {}
        for _,tag in pairs(pred) do
            table.insert(pred_str, rev_tag_dict[tag])
        end
        table.insert(lines, stringx.join(' ', pred_str))
    end
    utils.writefile(fname, stringx.join('\n', lines))
end

function experiment.do_experiment(config, para)
    -- para = {wordvec, tok_train, tag_train, tok_test, tag_test, valid_idx, model_file, pred_file}
    local word_dict, emb_table = read_data.read_emb(para.wordvec)
    config.V = #emb_table
    config.emb_table = emb_table

    local tag_dict = {}
    tag_dict['O'] = 1
    tag_dict['B'] = 2
    tag_dict['I'] = 3
    local seq_train = read_data.read_seq(para.tok_train, word_dict)
    seq_train = read_data.form_data_format(seq_train, config.V, config.window_size)
    local tag_train = read_data.read_tag(para.tag_train, tag_dict)
    local seq_test = read_data.read_seq(para.tok_test, word_dict)
    seq_test = read_data.form_data_format(seq_test, config.V, config.window_size)
    local tag_test = read_data.read_tag(para.tag_test, tag_dict)
    local rev_tag_dict = {}
    for v,p in pairs(tag_dict) do
        rev_tag_dict[p] = v
    end
    config.rev_tag_dict = rev_tag_dict
    local data
    if config.valid then
        local training, validing
        if #para.valid_idx == 0 then
            training, validing = utilities.split_dev_data({seq=seq_train, tag=tag_train}, 0.1)
        else
            training, validing = utilities.select_dev_data({seq=seq_train, tag=tag_train}, para.valid_idx)
        end
        data = {seq_train=training.seq, seq_dev=validing.seq, tag_train=training.tag, tag_dev=validing.tag}
    else
        data = {seq_train=seq_train, tag_train=tag_train}
    end
    data.seq_test = seq_test
    data.tag_test = tag_test
    local best_model, perf_valid, perf, perf_test = train_model.train(data, config)
    if para.model_file then
        torch.save(para.model_file, best_model)
    end
    --[[
    local result, predictions = train_model.cal_performance(best_model, {seq=seq_test, tag=tag_test}, config.rev_tag_dict)
    for v,p in pairs(result) do
        if type(p) == 'table' then
            print(v, p[3])
        else
            print(v, p)
        end
    end
    ]]
    print({perf_valid, perf, perf_test})
    experiment.save_result({perf_valid, perf, perf_test}, para.result_file)
    --experiment.save_predictions(predictions, tag_dict, para.pred_file)
end
