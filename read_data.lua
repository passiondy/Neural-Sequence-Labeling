require 'pl'
require 'torch'

read_data = {}
function read_data.read_emb(fname)
    local lines = utils.readlines(fname)
    local emb_table = {}
    local word_dict = {}
    local D = #(utils.split(lines[1]))-1
    local V = #lines
    for i=1,V do
        local vec = torch.zeros(D)
        local parts = utils.split(lines[i])
        local word = parts[1]
        word_dict[word] = i
        for j=1,D do
            vec[j] = tonumber(parts[j+1])
        end
        table.insert(emb_table, vec)
    end
    return word_dict, emb_table
end

function read_data.read_tag(fname, tag_dict)
    local lines = utils.readlines(fname)
    local idx = 0
	if not tag_dict then
		tag_dict = {}
	else
		for _,_ in pairs(tag_dict) do
			idx = idx+1
		end
	end
    local tags = {}
    for i=1,#lines do
        local parts = utils.split(lines[i])
        local tmp = {}
        for j=1,#parts do
            local t = parts[j]
            if not tag_dict[t] then
                idx = idx+1
                tag_dict[t] = idx
            end
            table.insert(tmp, tag_dict[t])
        end
        table.insert(tags, tmp)
    end
    return tags, tag_dict
end

function read_data.read_seq(fname, word_dict)
    local lines = utils.readlines(fname)
    local token_list = {}
    for i=1,#lines do
        local parts = utils.split(lines[i])
        local tmp = {}
        for j=1,#parts do
            local ID = word_dict[parts[j]]
            --table.insert(tmp, torch.Tensor{ID})
            table.insert(tmp, ID)
        end
        table.insert(token_list, tmp)
    end
    return token_list
end

function read_data.form_data_format(input_data, V, window_size)
	local BOS = V + 1
	local EOS = V + 1
	local formed_data = {}
	local offset = (window_size-1)/2
	local center = (window_size+1)/2
	for i=1,#input_data do
		local seq_in = input_data[i]
		local seq_out = {}
		for j=1,#seq_in do
			table.insert(seq_out, torch.zeros(window_size))
			seq_out[j][center] = seq_in[j]
			for k=1, offset do
				if j+k <= #seq_in then
					seq_out[j][center+k] = seq_in[j+k]
				else
					seq_out[j][center+k] = EOS
				end

				if j-k > 0 then
					seq_out[j][center-k] = seq_in[j-k]
				else
					seq_out[j][center-k] = BOS
				end
			end
		end
		table.insert(formed_data, seq_out)
	end
	return formed_data
end
