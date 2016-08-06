local CrfLabeler, parent = torch.class('CRF.CrfLabeler', 'nn.Module')

function CrfLabeler:__init(config)
    local luTable = nn.LookupTable(config.V+1, config.dim_in)
    local embedding = nn.Sequential()
    embedding:add(nn.Sequencer(luTable))
    local wSize = config.window_size
    if wSize > 1 then
        embedding:add(nn.Sequencer(nn.Reshape(1, wSize*config.dim_in)))
    end
    embedding:add(nn.JoinTable(1))
    -- initialize look up table
    for i=1, config.V do
        local vec = config.emb_table[i]
        luTable.weight[i]:copy(vec)
    end
    for i=1,1 do
        luTable.weight[config.V+i]:copy(torch.rand(config.dim_in):csub(0.5)*0.4)
    end
    self.embedding = embedding
    self.crf = CRF.crf(config)
end

function CrfLabeler:parameters()
    local para, dpara = {}, {}
    local p,dp = self.embedding:parameters()
    for i=1, #p do
        table.insert(para, p[i])
        table.insert(dpara, dp[i])
    end
    local p,dp = self.crf:parameters()
    for i=1, #p do
        table.insert(para, p[i])
        table.insert(dpara, dp[i])
    end
    return para, dpara
end

function CrfLabeler:predict(x)
    local vectors = self.embedding:forward(x)
    return self.crf:predict(vectors)
end

function CrfLabeler:forward(x, y)
    local vectors = self.embedding:forward(x)
    self.vectors = vectors
    return self.crf:forward(vectors, y)
end

function CrfLabeler:backward(x, y)
    local gradEmbed = self.crf:backward(self.vectors, y)
    self.embedding:backward(x, gradEmbed)
end

function CrfLabeler:zeroGradParameters()
    self.embedding:zeroGradParameters()
    self.crf:zeroGradParameters()
end
