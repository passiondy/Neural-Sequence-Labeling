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
end

function crf:forward(x, y)
    self:forwardPotential(x)
    local p = 0
    for i=1,x:size(1) do
        p = p + self.G[i]
        if i == 1 then
            p = p + self.initState[y[i]]
        else
            p = p + self.transition[y[i-1]][y[i]]
        end
    end
    p = -p+torch.log(self.Z)
    return p
end

function crf:logSumExp(M, dim)
    local max,_ = torch.max(M, dim)
    local tmp = M - torch.expand(max, M:size(1), M:size(2))
    return tmp:exp():sum(dim):reshape(self.K):log():add(max)
end

function crf:forwardPotential(x)
    local G = torch.mm(x, self.action:transpose(1,2)):exp()
    local F = torch.ones(x:size(1), self.K)
    F[1]:copy(torch.exp(self.initState + G[1]))
    local expTransition = torch.exp(self.transition)
    for i=2,x:size(1) do
        local tmp = torch.cmul(torch.expand(F:narrow(1, i-1, 1):transpose(1, 2), self.K, self.K), expTransition):sum(1)
        --local tmp = self:logSumExp(M, 1)
        F[i]:copy(torch.cmul(tmp, G[i]))
        tmp = torch.zeros(self.K)
        for k=1,self.K do
            local v = 0
            for t=1, self.K do
                v = v+F[i-1][t]*torch.exp(self.transition[t][k] + torch.dot(x[i], self.action[k]))
            end
            tmp[k] = v
        end
        print('f',i)
        print(F[i])
        print(tmp)
    end
    self.F = F
    self.G = G
    self.Z = F[F:size(1)]:sum()
    self.expTransition = expTransition
end

function crf:backwardPotential(x)
    local B = torch.ones(x:size(1), self.K)
    for i=x:size(1)-1,1,-1 do
        local tmp = torch.cmul(B[i+1], self.G[i+1]):reshape(1, self.K)
        B[i]:copy(torch.cmul(torch.expand(tmp, self.K, self.K), self.expTransition):sum(2))
        tmp = torch.zeros(self.K)
        for k=1,self.K do
            local v = 0
            for t=1,self.K do
                v = v+B[i+1][t]*torch.exp(self.transition[k][t] + torch.dot(x[i+1], self.action[t]))
            end
            tmp[k] = v
        end
        print('b', i)
        print(B[i])
        print(tmp)
    end
    self.B = B
end

function crf:backward(x, y)
    self:backwardPotential(x)
    local comp = torch.cmul(self.F, self.B):div(self.Z)
    local g = torch.mm(comp:transpose(1,2), x)
    local gradInput = torch.mm(comp, self.action)
    for i=1,x:size(1) do
        local y = y[i]
        g[y]:csub(x[i])
        gradInput[i]:csub(self.action[y])
    end
    self.gradAction:add(g)
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
    for i=1,x:size(1)-1 do
        local tmp = torch.cmul(self.G[i+1], self.B[i+1])
        local g = torch.cmul(torch.ger(self.F[i], tmp), self.expTransition)/self.Z
        local py = y[i]
        local y = y[i+1]
        g[py][y] = g[py][y] - 1
        self.gradTransition:add(g)
    end
    tmp = torch.zeros(self.gradTransition:size())
    for i=1, x:size(1)-1 do
        local py = y[i]
        local y = y[i+1]
        tmp[py][y] = tmp[py][y] - 1
        for k=1,self.K do
            for t=1,self.K do
                local p = self.F[i][k]*self.B[i+1][t]*torch.exp(self.transition[k][t] + torch.dot(x[i+1], self.action[t]))/self.Z
                tmp[k][t] = tmp[k][t] + p
            end
        end
    end
    print('gt', self.gradTransition)
    print('gtt', tmp)
    self.gradInitState:add(comp[1])
    self.gradInitState[y[1]] = self.gradInitState[y[1]] - 1
    tmp = torch.zeros(self.gradInitState:size())
    tmp[y[1]] = tmp[y[1]] - 1
    for k=1,self.K do
        local p = self.F[1][k]*self.B[1][k]/self.Z
        tmp[k] = tmp[k] + p
    end
    print('gi', self.gradInitState)
    print('git', tmp)
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
