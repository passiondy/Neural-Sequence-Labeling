require 'nn'
dofile 'CRF_beta.lua'
--dofile 'CRF.lua'
config = {K=5, D=8}
x = torch.rand(10,8)
y = torch.Tensor(10):random(1, 5)

crf = CRF.crf(config)
f = crf:forward(x,y)
print('---f----')
print(f)
print('-------')
g = crf:backward(x,y)
print('---g----')
print(g)
print('-------')
