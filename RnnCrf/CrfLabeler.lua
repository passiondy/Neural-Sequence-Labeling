local CrfLabeler, parent = torch.class('CRF.CrfLabeler', 'nn.Module')

function CrfLabeler:__init(config)
    self.rnn = nn.Sequential()
    self.rnn:add(build_rnn(config))
    self.rnn:add(nn.JoinTable(1))
    self.crf = CRF.crf(config)
end

function CrfLabeler:parameters()
    local para, dpara = {}, {}
    local p,dp = self.rnn:parameters()
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
    local vectors = self.rnn:forward(x)
    return self.crf:predict(vectors)
end

function CrfLabeler:forward(x, z)
    local vectors = self.rnn:forward(x)
    self.vectors = vectors
    local y = self.crf:predict_for_margin(vectors, z)
    self.y = y
    return self.crf:forward(vectors, y, z)
end

function CrfLabeler:backward(x, z)
    local gradEmbed = self.crf:backward(self.vectors, self.y, z)
    self.rnn:backward(x, gradEmbed)
end

function CrfLabeler:zeroGradParameters()
    self.rnn:zeroGradParameters()
    self.crf:zeroGradParameters()
end
