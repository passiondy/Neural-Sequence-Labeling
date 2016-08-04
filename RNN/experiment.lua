experiment = {}
function experiment.save_result(perf, fname)
    utils.writefile(fname, stringx.join('\n', perf))
end

function experiment.do_experiment(config, fold, wordvec_prefix, tok_prefix, tag_prefix, result_file)
	local word_dict, emb_table = read_data.read_emb(wordvec_prefix..'.wordvec')
	local local_config = tablex.deepcopy(config)
	local_config.V = #emb_table
	local_config.emb_table = emb_table
	print(tok_prefix..'.train.'..fold)

	local tag_dict = {}
	tag_dict['O'] = 1
	local seq_train = read_data.read_seq(tok_prefix..'.train.'..fold, word_dict)
	seq_train = read_data.form_data_format(seq_train, local_config.V, local_config.window_size)
	local tag_train = read_data.read_tag(tag_prefix..'.train.'..fold, tag_dict)
	local seq_test = read_data.read_seq(tok_prefix..'.test.'..fold, word_dict)
	seq_test = read_data.form_data_format(seq_test, local_config.V, local_config.window_size)
	local tag_test = read_data.read_tag(tag_prefix..'.test.'..fold, tag_dict)
	--local training, validing = utilities.split_dev_data({seq=seq_train, tag=tag_train}, 0.1)
	--local data = {seq_train=training.seq, seq_dev=validing.seq, tag_train=training.tag, tag_dev=validing.tag}
	local data = {seq_train=seq_train, seq_dev=seq_test, tag_train=tag_train, tag_dev=tag_test}
	local best_model = train_model.train(data, local_config)

	local result = train_model.cal_performance(best_model, {seq=seq_test, tag=tag_test})
    experiment.save_result(result, result_file)
end
