CRF = {}
local crf, parent = torch.class('CRF.crf', 'nn.Module')

function crf:__init(config)
    self.initState = torch.rand(config.K):csub(0.5)
    self.transition = torch.rand(config.K, config.K):csub(0.5)
    self.action = torch.rand(config.K, config.D):csub(0.5)
    self.K = config.K
    self.D = config.D
    self.gradInitState = torch.zeros(config.K)
    self.gradTransition = torch.zeros(config.K, config.K)
    self.gradAction = torch.zeros(config.K, config.D)
    if config.debug then
        self.debug = true
    else
        self.debug = false
    end
end

function crf:forward(x, y)
    self:forwardPotential(x)
    local p = 0
    for i=1,x:size(1) do
        p = p + self.G[i][y[i]]
        if i == 1 then
            p = p + self.initState[y[i]]
        else
            p = p + self.transition[y[i-1]][y[i]]
        end
    end
    p = -p+self.logZ
    return p
end

function crf:logSumExp(M, dim)
    if dim == 0 then
        local max = torch.max(M)
        return torch.log(torch.exp(M:csub(max)):sum())+max
    end
    local max,_ = torch.max(M, dim)
    M:csub(torch.expand(max, M:size(1), M:size(2)))
    return M:exp():sum(dim):reshape(self.K):log():add(max)
end

function crf:forwardPotential(x)
    local G = torch.mm(x, self.action:transpose(1,2))
    local logF = torch.zeros(x:size(1), self.K)
    logF[1]:copy(self.initState + G[1])
    for i=2,x:size(1) do
        local M = torch.add(torch.expand(logF[i-1]:reshape(self.K, 1), self.K, self.K), self.transition)
        local new = self:logSumExp(M, 1)new:add(G[i])
        logF[i]:copy(new)
        if self.debug then
            local tmp = torch.zeros(self.K)
            for k=1,self.K do
                local v = 0
                for t=1, self.K do
                    v = v+torch.exp(logF[i-1][t])*torch.exp(self.transition[t][k] + torch.dot(x[i], self.action[k]))
                end
                tmp[k] = v
            end
            print('f',i)
            print(logF[i])
            print(torch.log(tmp))
        end
    end
    self.logF = logF
    self.G = G
    self.logZ = self:logSumExp(logF[logF:size(1)], 0)
    if self.debug then print('Z-F', torch.exp(self.logZ)) end
end

function crf:backwardPotential(x)
    local logB = torch.zeros(x:size(1), self.K)
    for i=x:size(1)-1,1,-1 do
        local M = torch.add(logB[i+1], self.G[i+1]):reshape(1, self.K)
        local new = self:logSumExp(torch.add(torch.expand(M, self.K, self.K), self.transition), 2)
        logB[i]:copy(new)
        if self.debug then
            local tmp = torch.zeros(self.K)
            for k=1,self.K do
                local v = 0
                for t=1,self.K do
                    v = v+torch.exp(logB[i+1][t])*torch.exp(self.transition[k][t] + torch.dot(x[i+1], self.action[t]))
                end
                tmp[k] = v
            end
            print('b', i)
            print(logB[i])
            print(torch.log(tmp))
        end
    end
    self.logB = logB
    if self.debug then
        local z = self.logB[1] + self.G[1] + self.initState
        print('logZ-B', torch.exp(self:logSumExp(z, 0)))
    end
end

function crf:backward(x, y)
    self:backwardPotential(x)
    local comp = torch.add(self.logF, self.logB):csub(self.logZ):exp()
    local g = torch.mm(comp:transpose(1,2), x)
    local gradInput = torch.mm(comp, self.action)
    for i=1,x:size(1) do
        local y = y[i]
        g[y]:csub(x[i])
        gradInput[i]:csub(self.action[y])
    end
    self.gradAction:add(g)
    if self.debug then
        local tmp = torch.zeros(self.gradAction:size())
        for i=1,x:size(1) do
            local y = y[i]
            tmp[y]:csub(x[i])
            for k=1, self.K do
                tmp[k]:add(x[i]*comp[i][k])
            end
        end
        print('ga', self.gradAction)
        print('gat', tmp)
    end

    for i=1,x:size(1)-1 do
        local tmp = torch.add(self.G[i+1], self.logB[i+1])
        local g = (torch.expand(self.logF[i]:reshape(self.K, 1), self.K, self.K)+
            torch.expand(tmp:reshape(1, self.K), self.K, self.K) + self.transition):csub(self.logZ):exp()
        local py = y[i]
        local y = y[i+1]
        g[py][y] = g[py][y] - 1
        self.gradTransition:add(g)
    end
    if self.debug then
        tmp = torch.zeros(self.gradTransition:size())
        for i=1, x:size(1)-1 do
            local py = y[i]
            local y = y[i+1]
            tmp[py][y] = tmp[py][y] - 1
            for k=1,self.K do
                for t=1,self.K do
                    local p = self.logF[i][k] + self.logB[i+1][t] + self.transition[k][t] + torch.dot(x[i+1], self.action[t]) - self.logZ
                    tmp[k][t] = tmp[k][t] + torch.exp(p)
                end
            end
        end
        print('gt', self.gradTransition)
        print('gtt', tmp)
    end
    self.gradInitState:add(comp[1])
    self.gradInitState[y[1]] = self.gradInitState[y[1]] - 1
    if self.debug then
        tmp = torch.zeros(self.gradInitState:size())
        tmp[y[1]] = tmp[y[1]] - 1
        for k=1,self.K do
            local p = self.logF[1][k] + self.logB[1][k] - self.logZ
            tmp[k] = tmp[k] + torch.exp(p)
        end
        print('gi', self.gradInitState)
        print('git', tmp)
    end
    return gradInput
end

function crf:parameters()
    local para = {self.action, self.transition, self.initState}
    local dpara = {self.gradAction, self.gradTransition, self.gradInitState}
    return para, dpara
end

function crf:zeroGradParameters()
    self.gradAction:zero()
    self.gradTransition:zero()
    self.gradInitState:zero()
end
