torch.setnumthreads(1)
require 'rnn'
require 'torch'
require 'optim'
require 'xlua'
require 'pl'

dofile 'read_data.lua'
dofile 'build_model.lua'
dofile 'evaluate_func.lua'
dofile 'train_model.lua'
dofile 'experiment.lua'
dofile 'util_function.lua'

local config = {dim_in=300, activation='sigmoid', num_tag=3, maxIter=100, bSize= 100, dim_hidden=400,K=3}
config.window_size = 3
config.D = config.dim_hidden
config.optim_config={learningRate=0.01}
local direc = '/home/yingding/SequenceLabeling/RNN/data/'
local para = {}
--para = {wordvec, tok_train, tag_train, tok_test, tag_test, valid_idx, model_file, pred_file}
dataset = arg[1]
config.model = arg[2]
config.dim_out = tonumber(arg[3])
take = arg[4]

para.wordvec = direc..dataset..'.wordvec'
print(para.wordvec)
para.tok_train = direc..dataset..'.train.tok.exp'
para.tag_train = direc..dataset..'.train.tag.raw'
para.tok_test = direc..dataset..'.test.tok.exp'
para.tag_test = direc..dataset..'.test.tag.raw'
para.valid_idx = {}

suffix = config.model.. '_'.. config.dim_out ..'_'..take
para.result_file = 'result.'..dataset..'.'.. suffix
para.pred_file = 'predictions.'..dataset..'.'.. suffix
para.model_file = 'model.'..dataset..'.'..suffix
experiment.do_experiment(config, para)
