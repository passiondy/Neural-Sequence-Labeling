import os.path

def summarize(fout):
    output = []
    for data in ['laptop', 'restaurant-2014', 'restaurant-2015']:
        tmp = [data]
        for model in ['rnn', 'birnn', 'lstm', 'bilstm']:
            fscore_list = []
            for dim in [50, 100, 150, 200]:
                for t in range(1, 6):
                    fname = 'result.'+data+'.'+model+'_'+str(dim)+'_'+str(t)
                    if os.path.isfile(fname):
                        lines = open(fname).readlines()
                        fscore = float(lines[5])
                        perf = float(lines[10])
                        fscore_list.append((fscore, dim, perf, t))
            fscore_list.sort(key=lambda x:x[0], reverse=True)
            if fscore_list == []:
                continue
            tmp.append(model)
            tmp.append(str(fscore_list[0][1]))
            tmp.append(str(fscore_list[0][-1]))
            tmp.append(str(fscore_list[0][2]))
        output.append('\t'.join(tmp))

    open(fout, 'w').write('\n'.join(output))

def summarize_on_test(fout):
    output = []
    for data in ['laptop', 'restaurant-2014', 'restaurant-2015']:
        tmp = [data]
        for model in ['rnn', 'birnn', 'lstm', 'bilstm']:
            fscore_list = []
            for dim in [50, 100, 150, 200]:
                for t in range(1, 6):
                    fname = 'result.'+data+'.'+model+'_'+str(dim)+'_'+str(t)
                    if os.path.isfile(fname):
                        lines = open(fname).readlines()
                        perf = float(lines[18])
                        fscore_list.append((perf, dim, t))
            fscore_list.sort(key=lambda x:x[0], reverse=True)
            if fscore_list == []:
                continue
            tmp.append(model)
            tmp.append(str(fscore_list[0][1]))
            tmp.append(str(fscore_list[0][2]))
            tmp.append(str(fscore_list[0][0]))
        output.append('\t'.join(tmp))

    open(fout, 'w').write('\n'.join(output))

if __name__=='__main__':
    summarize('result.summ')
    summarize_on_test('result.summ.on_test')
