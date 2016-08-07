require 'nn'
dofile 'CRF.lua'
config = {K=5, D=8,debug=false}
x = torch.rand(10,8)
y = torch.Tensor(10):random(1, 5)

crf = CRF.crf(config)
for i=1,20 do
    f = crf:forward(x,y)
    g = crf:backward(x,y)
    print('---i----')
    print(f)
    --print(g)
    --[[
    print('-------')
    print('---g----')
    print('-------')
    ]]
    crf:updateParameters(0.01)
end
