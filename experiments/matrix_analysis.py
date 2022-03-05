import tqdm
from math import log
from itertools import product, chain, combinations
from multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from corpus_analysis.util import plot
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix, get_best_segments, smooth_matrix, peak_threshold
from corpus_analysis.structure.structure import matrix_to_labels
from corpus_analysis.structure.hierarchies import make_segments_hierarchical,\
    add_transitivity_to_matrix
from corpus_analysis.util import plot_matrix, load_json, save_json, multiprocess,\
    buffered_run
import salami
from experiments.matrix_ratings import matrix_rating_s, matrix_rating_b

BUFFER = 'salami/buffer/'

def analyze_matrix(params):
    index, value, param = params
    PARAMS[param] = value
    matrix = salami.load_fused_matrix(index)[0]
    l = np.mean([e[-1] for e in own_eval([index, 't', PARAMS])])
    mean = np.mean(matrix)
    meanlen = np.mean([len(s) for s in matrix_to_segments(matrix)])
    #ENTROPY!!!!!
    return [index, value, mean, meanlen, l]

#set up dataset for statistical matrix analysis
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

#statistical matrix analysis
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

def powerset(iterable):
    s = list(iterable)
    ps = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [np.array(list(e)) for e in ps]

def transitive_matrix(segs, target, params):
    m = make_segments_hierarchical(segs, params['MIN_LEN'],
        min_dist=params['MIN_DIST'], target=target, beta=params['BETA'],
        verbose=False)
    #m = salami.get_hierarchy()
    return m + m.T

def transitive_matrix_rating(args):
    index, beta, params = args
    params['BETA'] = beta
    matrix = salami.get_hierarchy_buf(index, params)[2]
    rating = matrix_rating_b(matrix)
    return rating, args

def sigma_matrix_ratings(args):
    index, sigma, params = args
    params['SIGMA'] = sigma
    smatrix, matrix = salami.get_matrices_buf(index, params)[:2]
    rating = matrix_rating_s(matrix)
    return matrix_rating_s(smatrix), matrix_rating_s(matrix), args

def beta_combi_experiment(index, params, plot_path, results=None):
    betas = np.array([0.2,0.3,0.4,0.5])
    
    smatrix, matrix, raw, beats = salami.get_matrices_buf(index, params)
    segs = matrix_to_segments(smatrix)
    target = raw
    
    transitives = []
    for b in betas:
        print(index, 'transitive', b)
        params['BETA'] = b
        transitives.append(transitive_matrix(segs, target, params))
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
        print(index, betas[c], combiratings[-1])
    best = matrixcombis[np.argmax(combiratings)]
    if results:
        data = read_and_prepare_results(results)
        print(index, 'baseline l', data[(data['SONG'] == index) & (data['METHOD'] == 'l')]['L'].iloc[0])
        print(index, 'baseline t', data[(data['SONG'] == index) & (data['METHOD'] == 't')]['L'].iloc[0])
        # print('baseline t', data[(data['SONG'] == index) & (data['MAX_GAP_RATIO'] == 0.2)
        #     & (data['SIGMA'] == 0.016) & (data['BETA'] == 0.6)]['L'].iloc[0])
    
    t = salami.labels_to_hierarchy(index, matrix_to_labels(best), best,
        salami.get_beats(index), salami.get_groundtruth(index))
    print(index, 'best', betacombis[np.argmax(combiratings)])
    return salami.evaluate(index, 'combi', list(params.values()), t[0], t[1])

def best_sigma(index, params, buffer=None, plot_path=None):#='salami/sigmas.json'):
    if buffer:
        sigmas = load_json(buffer)
        if sigmas and str(index) in sigmas:
            return sigmas[str(index)]
    #values = np.round(np.arange(0.001,0.011,0.001), 3)
    values = [0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,1.024]
    measures = []
    for v in values:
        params['SIGMA'] = v
        matrix = salami.load_fused_matrix(index, params, var_sigma=False)[0]
        # chroma = salami.get_beatwise_chroma(index)
        # chroma = MinMaxScaler().fit_transform(chroma)
        # matrix = 1-pairwise_distances(chroma, chroma, metric="cosine")
        # matrix = smooth_matrix(matrix, False, 2)
        # #matrix = matrix + matrix.T
        # matrix = peak_threshold(matrix, params['MEDIAN_LEN'], params['SIGMA'])
        # matrix = np.logical_or(matrix.T, matrix)
        # #matrix = salami.own_chroma_affinity(index)[1]#raw
        # #matrix = smooth_matrix(matrix, True, 1)
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

def prepare_and_log_results(results, params, thresh='SIGMA'):
    data = read_and_prepare_results(results)
    print("baseline:", data[data['METHOD'] == 'l']['L'].mean())
    #data = data[data['METHOD'] == 't']
    #data = data[data['BETA'] <= 0.5]
    #data = data[data['MAX_GAP_RATIO'] == 0.2]
    grouped = data[data['METHOD'] == 't'].groupby([thresh,'MAX_GAP_RATIO','BETA'])[['L']].mean()
    bestparams = grouped.idxmax().item()
    print("fixed:", grouped.max().item(), tuple(zip((thresh,'MAX_GAP_RATIO','BETA'), bestparams)))
    print("max:", np.mean(data.groupby(['SONG']).max()['L']))
    print("max var beta:", np.mean(data[(data[thresh] == bestparams[0])
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
    if isinstance(results, list):
        songs = sorted(results)
        results = None
    else:
        data, bestparams = prepare_and_log_results(results, params)
        songs = data['SONG'].unique()#[::20]
    
    # multiparams = [(s, params, plot_path, results) for s in songs]
    # evals = multiprocess('var sigma beta', var_sigma_beta2, multiparams, True)
    
    evals = [var_sigma_beta(s, params, plot_path, results) for s in songs]
    
    print('overall', np.mean([np.mean([r[-1] for r in rs]) for rs in evals]))

def beta_combi_experiment2(multiparams):
    return beta_combi_experiment(*multiparams)

def test_beta(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params)
    songs = data['SONG'].unique()#[::20]
    
    multiparams = [(s, params, plot_path, results) for s in songs]
    evals = multiprocess('var beta', beta_combi_experiment2, multiparams, True)
    
    #evals = [beta_combi_experiment(s, params, plot_path, results) for s in songs]
    
    print('overall', np.mean([np.mean([r[-1] for r in rs]) for rs in evals]))

def test_beta_measure(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params, 'THRESHOLD')
    data = data[(data['THRESHOLD'] == bestparams[0])
        & (data['MAX_GAP_RATIO'] == bestparams[1])
        & (data['SONG'].isin(data['SONG'].unique()[:]))
        ]
    print("check filter:", np.mean(data[(data['BETA'] == bestparams[2])]
        .groupby(['SONG']).max()['L']))
    
    params['THRESHOLD'] = bestparams[0]
    params['MAX_GAP_RATIO'] = bestparams[1]
    
    multiparams = [(i, b, params.copy()) for b in data['BETA'].unique()
        for i in data['SONG'].unique()]
    
    ratings = {}
    with Pool(processes=cpu_count()-2) as pool:
        for r in tqdm.tqdm(pool.imap_unordered(transitive_matrix_rating, multiparams),
                total=len(multiparams), desc='ratings'):
            index, beta, rating = r[1][0], r[1][1], r[0]
            #print(index, beta, rating)
            ratings[(index, beta)] = rating
    
    data['RATING'] = data.apply(lambda d: ratings[(d['SONG'], d['BETA'])], axis=1)
    
    plot(lambda: data.plot.scatter(x='RATING', y='L', c='SONG', colormap='tab20'), 'salami/RATINGS.png')
    print(data.loc[data.groupby('SONG')['RATING'].idxmax()]['L'].mean())

def test_sigma_measure(results, params, plot_path):
    data, bestparams = prepare_and_log_results(results, params, 'SIGMA')
    data = data[(data['MAX_GAP_RATIO'] == bestparams[1])
        & (data['SONG'].isin(data['SONG'].unique()[:10]))
        ]
    
    params['BETA'] = bestparams[2]
    params['MAX_GAP_RATIO'] = bestparams[1]
    
    multiparams = [(i, s, params.copy()) for s in data['SIGMA'].unique()
        for i in data['SONG'].unique()]
    
    ratings = {}
    sratings = {}
    with Pool(processes=cpu_count()-2) as pool:
        for r in tqdm.tqdm(pool.imap_unordered(sigma_matrix_ratings, multiparams),
                total=len(multiparams), desc='ratings', smoothing=0.1):
            srating, rating, (index, sigma, p) = r
            #print(index, beta, rating)
            ratings[(index, sigma)] = rating
            sratings[(index, sigma)] = srating
    
    data['RATING'] = data.apply(lambda d: ratings[(d['SONG'], d['SIGMA'])], axis=1)
    data['SRATING'] = data.apply(lambda d: sratings[(d['SONG'], d['SIGMA'])], axis=1)
    
    
    real_best = data.loc[data.groupby('SONG')['L'].idxmax()][['SONG','SIGMA','L']]
    # print(data[data['SONG'] == 108])
    print(real_best)
    
    fixedbeta = data[(data['BETA'] == bestparams[2])]
    plot(lambda: fixedbeta.plot.scatter(x='RATING', y='L', c='SONG', colormap='tab20'), 'salami/RATINGS_S.png')
    plot(lambda: fixedbeta.plot.scatter(x='SRATING', y='L', c='SONG', colormap='tab20'), 'salami/SRATINGS_S.png')
    
    bestsigmas_r = data.loc[data.groupby('SONG')['RATING'].idxmax()][['SONG','SIGMA']].reset_index()
    print(data.loc[data.groupby('SONG')['RATING'].idxmax()][['SONG','SIGMA','L']].reset_index())
    bestsigmas_r = data.merge(bestsigmas_r[['SONG','SIGMA']], how='inner')
    bestsigmas_rs = data.loc[data.groupby('SONG')['SRATING'].idxmax()][['SONG','SIGMA']].reset_index()
    bestsigmas_rs = data.merge(bestsigmas_rs[['SONG','SIGMA']], how='inner')
    r = fixedbeta.loc[fixedbeta.groupby('SONG')['RATING'].idxmax()]['L']
    rmax = bestsigmas_r.loc[bestsigmas_r.groupby('SONG')['L'].idxmax()]['L']
    rs = fixedbeta.loc[fixedbeta.groupby('SONG')['SRATING'].idxmax()]['L']
    rsmax = bestsigmas_rs.loc[bestsigmas_rs.groupby('SONG')['L'].idxmax()]['L']
    print('feature matrix rating', r.mean(), rmax.mean())
    print('segment matrix rating', rs.mean(), rsmax.mean())

#print(matrix_to_endpoint_vector(np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[0,0,1,1]])))