from math import log
from itertools import product, chain, combinations
import numpy as np
from corpus_analysis.util import plot
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix, get_best_segments
from corpus_analysis.structure.structure import matrix_to_labels
from corpus_analysis.structure.hierarchies import make_segments_hierarchical,\
    add_transitivity_to_matrix
from corpus_analysis.stats.util import entropy
from corpus_analysis.util import plot_matrix, flatten, load_json, save_json
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
        print('transitive', b)
        m = segments_to_matrix(make_segments_hierarchical(segs, params['MIN_LEN'],
            min_dist=params['MIN_DIST'], target=matrix, beta=b, verbose=False), matrix.shape)
        transitives.append(m + m.T)
    combis = powerset(np.arange(len(betas)))[1:]#ignore empty set
    betacombis = [betas[c] for c in combis]
    matrixcombis = []
    combiratings = []
    for c in combis:
        combi = np.logical_or.reduce([transitives[i] for i in c])
        combi = add_transitivity_to_matrix(combi)
        plot_matrix(combi, plot_path+'ccc'+str(index)+'-'+str(betas[c])+'.png')
        matrixcombis.append(combi)
        combiratings.append(matrix_rating(combi))
        print(betas[c], combiratings[-1])
    best = matrixcombis[np.argmax(combiratings)]
    print('baseline', data[(data['SONG'] == index) & (data['MAX_GAP_RATIO'] == 0.2)
        & (data['SIGMA'] == 0.016) & (data['BETA'] == 0.5)]['L'].iloc[0])
    t = salami.labels_to_hierarchy(matrix_to_labels(best), best,
        salami.get_beats(index), salami.get_groundtruth(index))
    print('best', betacombis[np.argmax(combiratings)])
    return salami.evaluate(index, 'combi', list(params.values()), t[0], t[1])

def powerset(iterable):
    s = list(iterable)
    ps = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [np.array(list(e)) for e in ps]

def distribution_measures(vectors, resolution=0):
    means = np.array([np.mean(v) for v in vectors])
    var_coeff = np.std(means)/np.mean(means)
    if resolution == 0: resolution = np.max([np.sum(v) for v in vectors])#full resolution
    means = np.round(means/np.max(means)*resolution).astype(int)
    return means, var_coeff, entropy(means)

def matrix_to_endpoint_vector(matrix):
    #ONLY UPPER/LOWER TRIANGLE!!?!???!!
    segs = matrix_to_segments(matrix)
    endpoints = flatten([[s[0], s[-1]] for s in segs if len(s) > 1], 1)
    endpoints += [s[0] for s in segs if len(s) == 1]
    return np.bincount([e[1] for e in endpoints])

def min_dist_between_nonzero(a):
    ix = np.where(a)[0]
    a[ix[:-1]] = np.diff(ix)
    a = a[:-1]
    return np.min(a[np.nonzero(a)])

def matrix_rating(matrix, resolution=0, minlen=10):
    #np.fill_diagonal(matrix, 0)
    if np.sum(matrix) == 0: return 0
    diagonals = to_diagonals(matrix)
    antidiagonals = to_diagonals(np.flip(matrix, axis=0))
    xmeans, xvar, xent = distribution_measures(matrix, resolution)
    dmeans, dvar, dent = distribution_measures(diagonals, resolution)
    admeans, advar, adent = distribution_measures(antidiagonals, resolution)
    nonzero = len([x for x in xmeans if np.sum(x) > 0]) / len(xmeans)
    decent = len([x for x in xmeans if 2 < np.sum(x) < 0.1*len(xmeans)]) / len(xmeans)
    segs = [len(s) for s in matrix_to_segments(matrix)]
    segs = [s for s in segs if s > 2]
    meanseglen, maxseglen, minseglen = np.mean(segs), np.max(segs), np.min(segs)
    pent = entropy(matrix_to_endpoint_vector(matrix))
    xdiffent = entropy(np.abs(np.diff(xmeans)))
    mindist = min_dist_between_nonzero(dmeans)
    #print(np.histogram(admeans)[0], np.histogram(dmeans)[0], np.histogram(xmeans)[0])
    #print(np.bincount(admeans))
    # print("s", meanseglen, maxseglen, nonzero)
    # print("e", entropy(admeans), entropy(xmeans))
    # print("v", advar, xvar)
    # print("*", entropy(admeans)*advar, entropy(xmeans)*xvar)
    # print("p", pent)
    #return adent*advar*xent*xvar if maxseglen >= minlen else 0
    #return xent*nonzero if maxseglen >= minlen else 0
    #return xent*xvar*nonzero if maxseglen >= minlen else 0 #0.511143729873462 0.5385528126486275
    #return xvar*nonzero if maxseglen >= minlen else 0 #0.5285681381755871 0.5486171025288525
    return xvar/xent*nonzero if maxseglen >= minlen else 0
    #return xvar*nonzero*maxseglen #0.5088370030634288 0.5312888863716688
    #return xvar*nonzero*meanseglen if maxseglen >= minlen else 0 #0.5173722584948066 0.5350565320835667
    #return nonzero/pent if maxseglen >= minlen else 0
    #return decent*xvar if maxseglen >= minlen else 0#xent*xvar
    #return nonzero/xent*xdiffent#if mindist > minlen else 0 #*log(len(segs))#/minseglen #if maxseglen >= minlen else 0

def best_sigma(index, params, buffer=None):#='salami/sigmas.json'):
    if buffer:
        sigmas = load_json(buffer)
        if sigmas and str(index) in sigmas:
            return sigmas[str(index)]
    #values = np.round(np.arange(0.001,0.011,0.001), 3)
    values = [0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128]#,0.256,0.512,1.024]
    measures = []
    for v in values:
        params['SIGMA'] = v
        matrix = salami.load_fused_matrix(index, params, var_sigma=False)[0]
        matrix = matrix + matrix.T
        measures.append(matrix_rating(matrix))
        print(index, v, measures[-1])
    best = values[np.argmax(measures)]
    if buffer:
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

def prepare_and_log_results(results, params):
    data = read_and_prepare_results(results)
    print("baseline:", data[data['METHOD'] == 'l']['L'].mean())
    data = data[data['METHOD'] == 't']
    #data = data[data['BETA'] <= 0.5]
    grouped = data.groupby(['SIGMA','MAX_GAP_RATIO','BETA'])[['L']].mean()
    bestparams = grouped.idxmax().item()
    print("fixed:", grouped.max().item(), tuple(zip(('SIGMA','MAX_GAP_RATIO','BETA'), bestparams)))
    print("max:", np.mean(data.groupby(['SONG']).max()['L']))
    print("max var beta:", np.mean(data[(data['SIGMA'] == bestparams[0])
        & (data['MAX_GAP_RATIO'] == bestparams[1])].groupby(['SONG']).max()['L']))
    # plot(lambda: data.boxplot(column=['L'], by=['SIGMA']))
    # plot(lambda: data.groupby(['SIGMA'])[['L']].mean().plot())
    # plot(lambda: data.groupby(['BETA'])[['L']].mean().plot())
    #plot(lambda: data.groupby(['MAX_GAP_RATIO'])[['L']].mean().plot())
    return data, bestparams

def test_beta_combi(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params)
    results = [beta_combi_experiment(s, params, plot_path, results)
        for s in data['SONG'].unique()]
    print('overall', np.mean([np.mean([r[-1] for r in rs]) for rs in results]))

def test_sigma(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params)
    songs = data['SONG'].unique()
    values = data['SIGMA'].unique()
    data = data[(data['MAX_GAP_RATIO'] == bestparams[1])]
    lmeasures = []
    lmaxes = []
    for s in songs:
        measures = []
        for v in values:
            params['SIGMA'] = v
            matrix = salami.load_fused_matrix(s, params, var_sigma=False)[0]
            matrix = matrix + matrix.T
            plot_matrix(matrix, plot_path+'eee'+str(s)+'-'+str(v)+'.png')
            rating = matrix_rating(matrix)
            print(s, v, rating, data[(data['SONG'] == s) & (data['SIGMA'] == v) & (data['BETA'] == bestparams[2])]['L'].iloc[0])
            measures.append(rating)
        best = values[np.argmax(measures)]
        l = data[(data['SONG'] == s) & (data['SIGMA'] == best) & (data['BETA'] == bestparams[2])]['L'].iloc[0]
        lmax = data[(data['SONG'] == s) & (data['SIGMA'] == best)].max()['L']
        print(s, np.argmax(measures), best, l, lmax, data[(data['SONG'] == s)].max()['L'])
        lmeasures.append(l)
        lmaxes.append(lmax)
    print(np.mean(lmeasures), np.mean(lmaxes))

def test_var_sigma_beta(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params)
    evals = []
    for s in data['SONG'].unique():
        params['SIGMA'] = best_sigma(s, params)
        print(s, 'best sigma', params['SIGMA'])
        evals.append(beta_combi_experiment(s, params, plot_path, results))
    print('overall', np.mean([np.mean([r[-1] for r in rs]) for rs in evals]))

#print(matrix_to_endpoint_vector(np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[0,0,1,1]])))