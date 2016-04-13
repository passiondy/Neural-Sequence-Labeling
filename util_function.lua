utilities = {}
function utilities.split_dev_data(data, ratio)
	local keys = {}
	for v,_ in pairs(data) do
		table.insert(keys, v)
	end
	local  N = #data[keys[1]]
	local idxList = torch.randperm(N)
	local dev_idxList = {}
	local train_idxList = {}
	for i=1, math.ceil(N*ratio) do
		table.insert(dev_idxList, idxList[i])
	end
	for i=math.ceil(N*ratio)+1, N do
		table.insert(train_idxList, idxList[i])
	end

	local train_data = {}
	local dev_data = {}
	for _,key in pairs(keys) do
		train_data[key] = {}
		dev_data[key] = {}
		for _,idx in pairs(train_idxList) do
			table.insert(train_data[key], data[key][idx])
		end
		for _,idx in pairs(dev_idxList) do
			table.insert(dev_data[key], data[key][idx])
		end
	end
	return train_data,dev_data
end
