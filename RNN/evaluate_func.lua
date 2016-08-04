evaluate_func = {}
function evaluate_func.evaluate_list(pred_tags, tags)
    local truth = 0
    local try = 0
    local get = 0

    local truth_token = 0
    local try_token = 0
    local get_token = 0

	local shoot = 0
	local miss = 0
    for i=1, #tags do
        if tags[i] == 2 then
            truth = truth + 1
        end
        if pred_tags[i] == 2 then
            try = try + 1
        end
    end
    local L = 0
    for i=1, #tags do
		-- token level accuracy
        if tags[i] == pred_tags[i] then
                shoot = shoot + 1
        else
                miss = miss + 1
        end
		-- token level target extraction precision/recall
        if tags[i] == 2 or tags[i] == 3 then
            truth_token = truth_token + 1
            if pred_tags[i] == 2 or pred_tags[i] == 3 then
                try_token = try_token + 1
                get_token = get_token + 1
            end
        else
            if pred_tags[i] == 2 or pred_tags[i] == 3 then
                try_token = try_token + 1
            end
        end
		-- segment level precision/recall
        if tags[i] == 1 then
            if pred_tags[i]~=3 and L == 1 then
                get = get + 1
            end
            L = 0
        elseif tags[i] == 2 then
            if L == 1 and pred_tags[i] ~= 3 then
                get = get+1
            end
            L = 0
            if pred_tags[i] == 2 then
                L = 1
            end
        elseif tags[i] == 3 then
            if pred_tags[i] ~= 3 then
                L = 0
            end
        end
		if L == 1 and i == #tags then
			get = get + 1
		end
    end

    return {try, truth, get, try_token, truth_token, get_token, shoot, miss}
end


function evaluate_func.get_prediction_tags(logsm_list)
    local pred_tags = {}
    local v,p
    for i, logsm in pairs(logsm_list) do
        if logsm:size():size() == 2 then
            v,p = torch.max(logsm, 2)
        else
            v,p = torch.max(logsm, 1)
        end
        local s = p:size()
        if s:size() == 2 then
            p = p[1][1]
        else
            p = p[1]
        end
        table.insert(pred_tags, p)
    end
    return pred_tags
end
