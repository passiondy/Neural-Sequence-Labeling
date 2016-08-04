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

local config = {dim_in=300, dim_out=150, activation='sigmoid', num_tag=3, maxIter=20, bSize= 1}
config.optim_config={learningRate=0.01}
config.window_size=3
local direc = 'restaurant-2014/'
local wordvec_prefix = direc..'restaurant-2014'
local tok_prefix = direc..'restaurant-2014.tok.processed'
local tag_prefix = direc..'restaurant-2014.tag.raw'
local result_file = 'result.restaurant-2014.'

local f = arg[1]
print('++++ Fold ', f, ' ++++')
experiment.do_experiment(config, f, wordvec_prefix, tok_prefix, tag_prefix, result_file..f)
