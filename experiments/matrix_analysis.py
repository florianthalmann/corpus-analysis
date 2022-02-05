from itertools import product, chain, combinations
import numpy as np
from corpus_analysis.util import plot
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix, get_best_segments
from corpus_analysis.structure.structure import matrix_to_labels
from corpus_analysis.structure.hierarchies import make_segments_hierarchical,\
    add_transitivity_to_matrix
from corpus_analysis.stats.util import entropy
from corpus_analysis.util import plot_matrix, flatten
import salami

def analyze_matrix(params):
    index, value, param = params
    PARAMS[param] = value
    matrix = salami.load_fused_matrix(index)[0]
    l = np.mean([e[-1] for e in own_eval([index, 't', PARAMS])])
    mean = np.mean(matrix)
    meanlen = np.mean([len(s) for s in matrix_to_segments(matrix)])
    #ENTROPY!!!!!
    return [index, value, mean, meanlen, l]

def matrix_analysis(songids):
    param, values = 'SIGMA', np.arange(0.006,0.01,0.001)
    #param, values = 'THRESHOLD', np.arange(0.2,4,0.2)
    params = [[i,v,param]
        for (v,i) in itertools.product(list(values), songids)]
    results = Data('salami/matricesH.csv',
        columns=['SONG', param, 'MEAN', 'MEANLEN', 'L'])
    with Pool(processes=cpu_count()-2) as pool:
        for r in tqdm.tqdm(pool.imap_unordered(analyze_matrix, params),
                total=len(params), desc='analyzing matrices'):
            results.add_rows([r])

def matrix_analysis2(file='salami/matricesH.csv'):
    path = 'salami/matrices'
    data = Data(file).read()
    songs = data['SONG'].unique()
    matrix_sizes = dict(zip(songs, [len(get_beats(s)) for s in songs]))
    plot(lambda: data.plot.scatter(x='MEAN', y='L'), path+'*ML-.pdf')
    plot(lambda: data.plot.scatter(x='MEANLEN', y='L'), path+'*MLL-.pdf')
    plot(lambda: data.plot.scatter(x='SIGMA', y='L'), path+'*SL-.pdf')
    
    plot(lambda: data.groupby(['SIGMA']).mean().plot(y='L'), path+'*-.pdf')
    
    # data = data.sort_values('L', ascending=False).drop_duplicates(['SONG'])
    # data['BEATS'] = data['SONG'].map(matrix_sizes)
    # print(data)
    # plot(lambda: data.plot.scatter(x='BEATS', y='L'), path+'L.pdf')
    # plot(lambda: data.plot.scatter(x='BEATS', y='SIGMA'), path+'SIGMA.pdf')
    # plot(lambda: data.plot.scatter(x='BEATS', y='MEAN'), path+'MEAN.pdf')
    # plot(lambda: data.plot.scatter(x='BEATS', y='MEANLEN'), path+'MEANLEN.pdf')
    # plot(lambda: data.plot.scatter(x='MEANLEN', y='L'), path+'MLL.pdf')
    # plot(lambda: data.plot.scatter(x='SIGMA', y='MEANLEN'), path+'SML.pdf')
    # plot(lambda: data.plot.scatter(x='SIGMA', y='MEAN'), path+'SM.pdf')
    
    data = Data(file).read()
    data['MLM'] = data['MEANLEN'] / data['MEAN']
    for s in songs:
        data2 = data[data['SONG'] == s]
        plot(lambda: data2.plot.scatter(x='SIGMA', y='L'), path+'-SIGMA'+str(s)+'-.pdf')
        # plot(lambda: data2.plot.scatter(x='MEAN', y='L'), path+'-ML'+str(s)+'.pdf')
        # plot(lambda: data2.plot.scatter(x='MEANLEN', y='L'), path+'-MLL'+str(s)+'.pdf')
        # plot(lambda: data2.plot.scatter(x='MLM', y='L'), path+'-MLML'+str(s)+'.pdf')

def entropy_experiment(index, params, plot_path, results, resolution=10, minlen=8):
    data = read_and_prepare_results(results)
    #param, values = 'SIGMA', np.round(np.arange(0.001,0.011,0.001), 3)
    #param, values = 'SIGMA', np.round(np.arange(0.01,0.11,0.01), 2)
    #param, values = 'SIGMA', [0.0001,0.001,0.005,0.01,0.1,1,10]
    param, values = 'SIGMA', [0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128]#,0.256,0.512,1.024]
    #param, values = 'BETA', [0.2,0.3,0.4,0.5]
    if param == 'BETA':
        data = data[(data['MAX_GAP_RATIO'] == 0.2) & (data['SIGMA'] == 0.016)]
    else:
        data = data[(data['MAX_GAP_RATIO'] == 0.2) & (data['BETA'] == 0.5)]
    matrix = None
    for v in values:
        params[param] = v
        m = salami.load_fused_matrix(index, params, var_sigma=False)[0]
        if param == 'BETA':
            segs = matrix_to_segments(get_best_segments(m, params['MIN_LEN'],
                min_dist=params['MIN_DIST'], min_val=1-params['MAX_GAP_RATIO'],
                max_gap_len=params['MAX_GAPS']))
            m = segments_to_matrix(make_segments_hierarchical(segs, params['MIN_LEN'],
                min_dist=params['MIN_DIST'], target=m, beta=params['BETA'], verbose=False), m.shape)
        m = m + m.T
        if param == 'BETA' and matrix is not None:
            matrix = add_transitivity_to_matrix(np.logical_or(matrix, m))
        else: matrix = m
        plot_matrix(matrix, plot_path+'eee'+str(index)+'-'+str(v)+'.png')
        #plot_matrix(blockmodel(matrix), plot_path+'e'+str(index)+'-'+str(v)+'*.png')
        
        print(v)
        rating = matrix_rating(matrix)
        print(rating)
        print(data[(data['SONG'] == index) & (data[param] == v)]['L'].iloc[0])
        
        if param == 'BETA':
            t = salami.labels_to_hierarchy(matrix_to_labels(matrix), matrix,
                salami.get_beats(index), salami.get_groundtruth(index))
            salami.evaluate(index, param, list(params.values()), t[0], t[1])
        #np.std(xmeans)/np.mean(xmeans), entropy(xmeans)*var_coeff)
        #print(fractal_dimension(matrix))

def beta_combi_experiment(index, params, plot_path, results):
    data = read_and_prepare_results(results)
    betas = np.array([0.2,0.3,0.4,0.5])
    matrix = salami.load_fused_matrix(index, params, var_sigma=False)[0]
    segs = matrix_to_segments(get_best_segments(matrix, params['MIN_LEN'],
        min_dist=params['MIN_DIST'], min_val=1-params['MAX_GAP_RATIO'],
        max_gap_len=params['MAX_GAPS']))
    transitives = []
    for b in betas:
        m = segments_to_matrix(make_segments_hierarchical(segs, params['MIN_LEN'],
            min_dist=params['MIN_DIST'], target=matrix, beta=b, verbose=False), matrix.shape)
        transitives.append(m + m.T)
    combis = powerset(np.arange(len(betas)))[1:]#ignore empty set
    print(combis)
    betacombis = [betas[c] for c in combis]
    matrixcombis = []
    combiratings = []
    for c in combis:
        combi = np.logical_or.reduce([transitives[i] for i in c])
        combi = add_transitivity_to_matrix(combi)
        plot_matrix(combi, plot_path+'ccc'+str(index)+'-'+str(list(c))+'.png')
        matrixcombis.append(combi)
        combiratings.append(matrix_rating(combi))
        print(betas[c], combiratings[-1])
    best = matrixcombis[np.argmax(combiratings)]
    print('baseline', data[(data['SONG'] == index) & (data['MAX_GAP_RATIO'] == 0.2)
        & (data['SIGMA'] == 0.016) & (data['BETA'] == 0.5)]['L'].iloc[0])
    t = salami.labels_to_hierarchy(matrix_to_labels(best), best,
        salami.get_beats(index), salami.get_groundtruth(index))
    print('best', betacombis[np.argmax(combiratings)])
    salami.evaluate(index, 'combi', list(params.values()), t[0], t[1])

def powerset(iterable):
    s = list(iterable)
    ps = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [np.array(list(e)) for e in ps]

def entropy_x_variation(vectors, var_weight=1):
    means = np.array([np.mean(v) for v in vectors])
    var_coeff = np.std(means)/np.mean(means)
    max = np.max([np.sum(v) for v in vectors])
    means = np.round(means/np.max(means)*max).astype(int)
    return entropy(means)*(var_coeff**var_weight)

def matrix_to_endpoint_vector(matrix):
    #ONLY UPPER/LOWER TRIANGLE!!?!???!!
    segs = matrix_to_segments(matrix)
    endpoints = flatten([[s[0], s[-1]] for s in segs if len(s) > 1], 1)
    endpoints += [s[0] for s in segs if len(s) == 1]
    return np.bincount([e[1] for e in endpoints])

def matrix_rating(matrix, minlen=10):
    if np.sum(matrix) == 0: return 0
    antidiagonals = to_diagonals(np.flip(matrix, axis=0))
    admeans = np.array([np.mean(d) for d in antidiagonals])
    admax = np.max([np.sum(d) for d in antidiagonals])
    admeans = np.round(admeans/np.max(admeans)*admax).astype(int)
    xmeans = np.sum(matrix, axis=0)
    xmeans = np.round(xmeans).astype(int)
    nonzero = len([x for x in xmeans if np.sum(x) > 0]) / len(xmeans)
    segs = [len(s) for s in matrix_to_segments(matrix)]
    #segs = [s for s in segs if s > 2]
    meanseglen, maxseglen = np.mean(segs), np.max(segs)
    xvar = np.std(xmeans)/np.mean(xmeans)
    advar = np.std(admeans)/np.mean(admeans)
    pent = entropy(matrix_to_endpoint_vector(matrix))
    #print(np.histogram(admeans)[0], np.histogram(dmeans)[0], np.histogram(xmeans)[0])
    #print(np.bincount(xmeans))
    #print(np.bincount(admeans))
    # print("s", meanseglen, maxseglen, nonzero)
    # print("e", entropy(admeans), entropy(xmeans))
    # print("v", advar, xvar)
    # print("*", entropy(admeans)*advar, entropy(xmeans)*xvar)
    # print("p", pent)
    #return entropy_x_variation(antidiagonals)*entropy_x_variation(matrix) if maxseglen >= minlen else 0
    #return entropy(xmeans)*nonzero if maxseglen >= minlen else 0
    #return entropy_x_variation(matrix)*nonzero if maxseglen >= minlen else 0 #0.511143729873462 0.5385528126486275
    return xvar*nonzero if maxseglen >= minlen else 0 #0.5285681381755871 0.5486171025288525
    #return xvar*nonzero*maxseglen #0.5088370030634288 0.5312888863716688
    #return xvar*nonzero*meanseglen if maxseglen >= minlen else 0 #0.5173722584948066 0.5350565320835667
    #return nonzero/pent if maxseglen >= minlen else 0

def test_peak_param(file='salami/matricesH.csv', resolution=10, minlen=10):
    param = 'SIGMA'
    data = Data(file).read()
    songs = data['SONG'].unique()
    values = data[param].unique()
    print(np.mean(data[data[param] == 0.005]['L']))
    print(np.mean(data.groupby(['SONG']).max()['L']))
    lmeasures = []
    for s in songs:
        measures = []
        for v in values:
            PARAMS[param] = v
            matrix = salami.load_fused_matrix(s, var_sigma=False)[0]
            matrix = matrix + matrix.T
            # xm = np.mean(matrix, axis=0)
            # entrop = entropy(np.round(xm/np.max(xm)*resolution).astype(int))
            antidiagonals = to_diagonals(np.flip(matrix, axis=0))
            [len(s) for s in matrix_to_segments(matrix)]
            seglenmax = np.max([len(s) for s in matrix_to_segments(matrix)])
            m = entropy_x_variation(antidiagonals) * entropy_x_variation(matrix) \
                if seglenmax >= minlen else 0
            #seglenmean = np.mean([len(s) for s in matrix_to_segments(matrix)])
            measures.append(m)#*seglenmean)
        best = values[np.argmax(measures)]
        l = data[(data['SONG'] == s) & (data['SIGMA'] == best)]['L'].iloc[0]
        print(s, np.argmax(measures), best, l, data[(data['SONG'] == s)].max()['L'])
        lmeasures.append(l)
    print(np.mean(lmeasures))

def best_sigma(index, buffer='salami/sigmas.json', minmaxlen=12):
    sigmas = load_json(buffer)
    if sigmas and str(index) in sigmas:
        return sigmas[str(index)]
    #values = np.round(np.arange(0.001,0.011,0.001), 3)
    values = [0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128]#,0.256,0.512,1.024]
    measures = []
    for v in values:
        PARAMS['SIGMA'] = v
        matrix = salami.load_fused_matrix(index, var_sigma=False)[0]
        matrix = matrix + matrix.T
        antidiagonals = to_diagonals(np.flip(matrix, axis=0))
        [len(s) for s in matrix_to_segments(matrix)]
        seglenmax = np.max([len(s) for s in matrix_to_segments(matrix)])
        m = entropy_x_variation(antidiagonals) * entropy_x_variation(matrix) \
            if seglenmax >= minmaxlen else 0
        measures.append(m)
    best = values[np.argmax(measures)]
    sigmas = load_json(buffer)
    sigmas = sigmas if sigmas else {}
    sigmas[index] = best
    save_json(buffer, sigmas)
    return best

def allbut(columns, exclude):
    return [item for item in columns if item not in exclude]

def read_and_prepare_results(results):
    data = results.read()
    #average cases with multiple groundtruths
    return data.groupby(allbut(data.columns, ['REF','P','R','L'])).mean().reset_index()

def test_measure(results, params):
    data = read_and_prepare_results(results)
    print("baseline:", data[data['METHOD'] == 'l']['L'].mean())
    data = data[data['METHOD'] == 't']
    #data = data[data['BETA'] <= 0.5]
    grouped = data.groupby(['SIGMA','MAX_GAP_RATIO','BETA'])[['L']].mean()
    bestcombi = grouped.idxmax().item()
    print("fixed:", grouped.max().item(), tuple(zip(('SIGMA','MAX_GAP_RATIO','BETA'), bestcombi)))
    print("max:", np.mean(data.groupby(['SONG']).max()['L']))
    print("max var beta:", np.mean(data[(data['SIGMA'] == bestcombi[0]) & (data['MAX_GAP_RATIO'] == bestcombi[1])].groupby(['SONG']).max()['L']))
    # plot(lambda: data.boxplot(column=['L'], by=['SIGMA']))
    # plot(lambda: data.groupby(['SIGMA'])[['L']].mean().plot())
    # plot(lambda: data.groupby(['BETA'])[['L']].mean().plot())
    #plot(lambda: data.groupby(['MAX_GAP_RATIO'])[['L']].mean().plot())
    
    songs = data['SONG'].unique()
    values = data['SIGMA'].unique()
    data = data[(data['MAX_GAP_RATIO'] == bestcombi[1])]
    lmeasures = []
    lmaxes = []
    for s in songs:
        measures = []
        for v in values:
            params['SIGMA'] = v
            matrix = salami.load_fused_matrix(s, params, var_sigma=False)[0]
            matrix = matrix + matrix.T
            rating = matrix_rating(matrix)
            print(s, v, rating, data[(data['SONG'] == s) & (data['SIGMA'] == v) & (data['BETA'] == bestcombi[2])]['L'].iloc[0])
            measures.append(rating)
        best = values[np.argmax(measures)]
        l = data[(data['SONG'] == s) & (data['SIGMA'] == best) & (data['BETA'] == bestcombi[2])]['L'].iloc[0]
        lmax = data[(data['SONG'] == s) & (data['SIGMA'] == best)].max()['L']
        print(s, np.argmax(measures), best, l, lmax, data[(data['SONG'] == s)].max()['L'])
        lmeasures.append(l)
        lmaxes.append(lmax)
    print(np.mean(lmeasures), np.mean(lmaxes))

#print(matrix_to_endpoint_vector(np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[0,0,1,1]])))