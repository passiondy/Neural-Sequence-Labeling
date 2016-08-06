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

local config = {dim_in=300, activation='sigmoid', num_tag=3, maxIter=2, bSize= 1}
config.window_size = 3
config.optim_config={learningRate=0.01}
local direc = '/home/yingding/SequenceLabeling/data/'
local para = {}
--para = {wordvec, tok_train, tag_train, tok_test, tag_test, valid_idx, model_file, pred_file}
para.wordvec = direc..'restaurant-2014.wordvec'
para.tok_train = direc..'restaurant-2014.train.tok.exp'
para.tag_train = direc..'restaurant-2014.train.tag.raw'
para.tok_test = direc..'restaurant-2014.test.tok.exp'
para.tag_test = direc..'restaurant-2014.test.tag.raw'
para.valid_idx = {}

config.model = arg[1]
config.dim_out = tonumber(arg[2])
take = arg[3]
suffix = config.model.. '_'.. config.dim_out ..'_'..take
para.result_file = 'result.restaurant-2014.'.. suffix
para.pred_file = 'predictions.restaurant-2014.'.. suffix
para.model_file = 'model.restaurant-2014.'..suffix
experiment.do_experiment(config, para)
