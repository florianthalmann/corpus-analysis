from math import log
from itertools import product, chain, combinations
import numpy as np
from corpus_analysis.util import plot
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix, get_best_segments
from corpus_analysis.alignment.util import strided2D
from corpus_analysis.structure.structure import matrix_to_labels
from corpus_analysis.structure.hierarchies import make_segments_hierarchical,\
    add_transitivity_to_matrix
from corpus_analysis.util import plot_matrix, load_json, save_json, multiprocess
import salami
from experiments.matrix_ratings import matrix_rating_s, matrix_rating_b

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
        rating = matrix_rating_s(matrix)
        print(rating)
        print(data[(data['SONG'] == index) & (data[param] == v)]['L'].iloc[0])
        
        if param == 'BETA':
            t = salami.labels_to_hierarchy(index, matrix_to_labels(matrix), matrix,
                salami.get_beats(index), salami.get_groundtruth(index))
            salami.evaluate(index, param, list(params.values()), t[0], t[1])
        #np.std(xmeans)/np.mean(xmeans), entropy(xmeans)*var_coeff)
        #print(fractal_dimension(matrix))

def beta_combi_experiment(index, params, plot_path, results=None):
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
        combiratings.append(matrix_rating_b(combi))
        print(betas[c], combiratings[-1])
    best = matrixcombis[np.argmax(combiratings)]
    if results:
        data = read_and_prepare_results(results)
        print('baseline l', data[(data['SONG'] == index) & (data['METHOD'] == 'l')]['L'].iloc[0])
        print('baseline t', data[(data['SONG'] == index) & (data['MAX_GAP_RATIO'] == 0.2)
            & (data['SIGMA'] == 0.016) & (data['BETA'] == 0.6)]['L'].iloc[0])
    
    t = salami.labels_to_hierarchy(index, matrix_to_labels(best), best,
        salami.get_beats(index), salami.get_groundtruth(index))
    print('best', betacombis[np.argmax(combiratings)])
    return salami.evaluate(index, 'combi', list(params.values()), t[0], t[1])

def powerset(iterable):
    s = list(iterable)
    ps = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [np.array(list(e)) for e in ps]

def best_sigma(index, params, buffer=None, plot_path=None):#='salami/sigmas.json'):
    if buffer:
        sigmas = load_json(buffer)
        if sigmas and str(index) in sigmas:
            return sigmas[str(index)]
    #values = np.round(np.arange(0.001,0.011,0.001), 3)
    values = [0.001,0.002,0.004,0.008,0.016,0.032,0.064]#,0.128]#,0.256,0.512,1.024]
    measures = []
    for v in values:
        params['SIGMA'] = v
        matrix = salami.load_fused_matrix(index, params, var_sigma=False)[0]
        matrix = matrix + matrix.T
        measures.append(matrix_rating_s(matrix))
        if plot_path: plot_matrix(matrix, plot_path+'eee'+str(index)+'-'+str(v)+'.png')
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
    #data = data[data['METHOD'] == 't']
    #data = data[data['BETA'] <= 0.5]
    #data = data[data['MAX_GAP_RATIO'] == 0.2]
    grouped = data[data['METHOD'] == 't'].groupby(['SIGMA','MAX_GAP_RATIO','BETA'])[['L']].mean()
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
    values = data[data['METHOD'] == 't']['SIGMA'].unique()#[:-1]
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
            rating = matrix_rating_s(matrix)
            print(s, v, rating, data[(data['SONG'] == s) & (data['SIGMA'] == v) & (data['BETA'] == bestparams[2])]['L'].iloc[0])
            measures.append(rating)
        # bb = np.argwhere(np.array(measures) <= 0.2)
        # best = values[bb[0][0] if len(bb) > 0 else -1]
        best = values[np.argmax(measures)]
        l = data[(data['SONG'] == s) & (data['SIGMA'] == best) & (data['BETA'] == bestparams[2])]['L'].iloc[0]
        lmax = data[(data['SONG'] == s) & (data['SIGMA'] == best)].max()['L']
        print(s, np.argmax(measures), best, l, lmax, data[(data['METHOD'] == 't') & (data['SONG'] == s)].max()['L'])
        lmeasures.append(l)
        lmaxes.append(lmax)
    print(np.mean(lmeasures), np.mean(lmaxes))

def var_sigma_beta(index, params, plot_path, results=None):
    params['SIGMA'] = best_sigma(index, params, plot_path=plot_path)
    print(index, 'best sigma', params['SIGMA'])
    return beta_combi_experiment(index, params, plot_path, results)

def var_sigma_beta2(multiparams):
    return var_sigma_beta(*multiparams)

def test_var_sigma_beta(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params)
    
    multiparams = [(s, params, plot_path, results) for s in data['SONG'].unique()]
    evals = multiprocess('var sigma beta', var_sigma_beta2, multiparams, True)
    
    # evals = [var_sigma_beta(s, params, plot_path, results)
    #     for s in data['SONG'].unique()]
    
    print('overall', np.mean([np.mean([r[-1] for r in rs]) for rs in evals]))

#print(matrix_to_endpoint_vector(np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[0,0,1,1]])))