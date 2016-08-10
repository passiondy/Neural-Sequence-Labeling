require 'rnn'
dofile 'CRF.lua'
dofile 'CrfLabeler.lua'

function build_model(config)
    return CRF.CrfLabeler(config)
end
