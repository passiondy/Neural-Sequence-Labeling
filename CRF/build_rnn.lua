require 'rnn'

function build_rnn(config)
    local luTable = nn.LookupTable(config.V+1, config.dim_in)
    local model = nn.Sequential()
    local wSize = config.window_size
    model:add(nn.Sequencer(luTable))
    if wSize > 1 then
        model:add(nn.Sequencer(nn.Reshape(1, wSize*config.dim_in)))
    end
    local fwd
    if config.activation == 'tanh' then
        fwd = nn.Recurrent(config.dim_out, nn.Linear(wSize*config.dim_in, config.dim_out), nn.Linear(config.dim_out, config.dim_out), nn.Tanh())
    else
        fwd = nn.Recurrent(config.dim_out, nn.Linear(wSize*config.dim_in, config.dim_out), nn.Linear(config.dim_out, config.dim_out), nn.Sigmoid())
    end
    model:add(nn.Sequencer(fwd))
    model:add(nn.Sequencer(nn.Linear(config.dim_out, config.num_tag)))
    model:add(nn.Sequencer(nn.LogSoftMax()))
    local para, dpara = model:parameters()
    for i=1,#para do
        para[i]:copy(torch.rand(para[i]:size()):csub(0.5)*0.4)
    end
    -- initialize look up table
    for i=1, config.V do
        local vec = config.emb_table[i]
        luTable.weight[i]:copy(vec)
    end
    for i=1,1 do
        luTable.weight[config.V+i]:copy(torch.rand(config.dim_in):csub(0.5)*0.4)
    end
    --
    return model
end
