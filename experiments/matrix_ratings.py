import numpy as np
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix
from corpus_analysis.util import plot_matrix, flatten, summarize_matrix
from corpus_analysis.stats.util import entropy

def matrix_rating_s(matrix, resolution=10, minlen=9):
    #np.fill_diagonal(matrix, 0)
    if np.sum(matrix) == 0: return 0
    #matrix = segments_to_matrix([s for s in matrix_to_segments(matrix) if len(s) > 1])
    # diagonals = to_diagonals(matrix)
    # antidiagonals = to_diagonals(np.flip(matrix, axis=0))
    xmeans, xvar, xent = distribution_measures(matrix, resolution)
    # dmeans, dvar, dent = distribution_measures(diagonals, resolution)
    # admeans, advar, adent = distribution_measures(antidiagonals, resolution)
    nonzero = len([x for x in xmeans if np.sum(x) > 0]) / len(xmeans)
    decent = len([x for x in xmeans if 2 < np.sum(x) < 0.1*len(xmeans)]) / len(xmeans)
    segs = [len(s) for s in matrix_to_segments(matrix)]
    ones = (len([s for s in segs if s < round(matrix.shape[0]/50)])+1)/len(segs)
    segs = [s for s in segs if 4 < s < 10]
    segs = [0] if len(segs) == 0 else segs
    meanseglen, maxseglen, minseglen = np.mean(segs), np.max(segs), np.min(segs)
    # pent = entropy(matrix_to_endpoint_vector(matrix))
    # mindist = min_dist_between_nonzero(dmeans)
    # xdiff = np.abs(np.diff(xmeans))
    # xdiffent = entropy(xdiff)
    # xdiffvar = np.std(xdiff)/np.mean(xdiff)
    # addiff = np.abs(np.diff(admeans))
    # addiffent = entropy(addiff)
    # addiffvar = np.std(addiff)/np.mean(addiff)
    # dotprop = len([s for s in segs if s == 1])/len(segs)
    #windows = np.concatenate(strided2D(matrix, 5))
    #print(len(windows), windows[0])
    # summary = summarize_matrix(matrix.astype(int), 50)
    # #maxsummary = np.percentile(summary, 75)#np.std(smoothed)#/np.mean(smoothed)
    # maxsummary = np.max(summary)
    #print(np.histogram(admeans)[0], np.histogram(dmeans)[0], np.histogram(xmeans)[0])
    #print(np.bincount(admeans))
    # print("s", meanseglen, maxseglen, nonzero)
    # print("e", entropy(admeans), entropy(xmeans))
    # print("v", advar, xvar)
    # print("*", entropy(admeans)*advar, entropy(xmeans)*xvar)
    # print("p", pent)
    #return xent*nonzero if maxseglen >= minlen else 0
    #return meanseglen*nonzero*xvar if maxseglen >= minlen else 0#4<s<10, 0.540913093460213 0.5605342014132532
    #return meanseglen*nonzero*xvar if maxseglen >= minlen else 0 #0.5298230209053344 0.5513507689993286
    #return np.max(summary) #(20, 0.3) 0.5183077935270706 0.5488571088790347
    #return xvar*nonzero if maxseglen >= minlen else 0 #0.5285681381755871 0.5486171025288525
    #return np.max(summary) #(10, 0.4) 0.5118378874621198 0.5431795285615351
    #return xvar*xdiffvar*nonzero if maxseglen >= minlen else 0 #0.528124128808957 0.5435334779559963
    #return xvar*decent if maxseglen >= minlen else 0 #0.5254184248747081 0.5406039283674171
    #return meanseglen #if maxseglen >= minlen else 0 #0.5061975025699631 0.5391262216955992
    #return xent*xvar*nonzero if maxseglen >= minlen else 0 #0.511143729873462 0.5385528126486275
    #return xvar*xdiffent*nonzero if maxseglen >= minlen else 0 #0.5004686802145721 0.5379346988947449
    #return adent*advar*xent*xvar if maxseglen >= minlen else 0 #0.5027575872168802 0.5355081591763832
    #return xvar*nonzero*meanseglen if maxseglen >= minlen else 0 #0.5173722584948066 0.5350565320835667
    #return xvar*nonzero*maxseglen #0.5088370030634288 0.5312888863716688
    #return xdiffvar*nonzero if maxseglen >= minlen else 0 #0.47609386764634976 0.5217101857451909 !!
    #return meanseglen*nonzero*xvar if maxseglen >= minlen else 0#*nonzero*xvar if maxseglen >= minlen else 0 #0.5298230209053344 0.5513507689993286
    return nonzero/ones*xvar
    #return  if maxseglen >= minlen else 0 #6
    #return nonzero/pent if maxseglen >= minlen else 0
    #return decent*xvar if maxseglen >= minlen else 0#xent*xvar
    #return nonzero/xent*xdiffent#if mindist > minlen else 0 #*log(len(segs))#/minseglen #if maxseglen >= minlen else 0

def matrix_rating_b(matrix, resolution=10, minlen=10):
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
    segs = [s for s in segs if s > 4]
    segs = [0] if len(segs) == 0 else segs
    meanseglen, maxseglen, minseglen = np.mean(segs), np.max(segs), np.min(segs)
    pent = entropy(matrix_to_endpoint_vector(matrix))
    mindist = min_dist_between_nonzero(dmeans)
    xdiff = np.abs(np.diff(xmeans))
    xdiffent = entropy(xdiff)
    xdiffvar = np.std(xdiff)/np.mean(xdiff)
    addiff = np.abs(np.diff(admeans))
    addiffent = entropy(addiff)
    addiffvar = np.std(addiff)/np.mean(addiff)
    dotprop = len([s for s in segs if s == 1])/len(segs)
    #windows = np.concatenate(strided2D(matrix, 5))
    #print(len(windows), windows[0])
    # summary = summarize_matrix(matrix.astype(int), 50)
    # #maxsummary = np.percentile(summary, 75)#np.std(smoothed)#/np.mean(smoothed)
    # maxsummary = np.max(summary)
    #print(np.histogram(admeans)[0], np.histogram(dmeans)[0], np.histogram(xmeans)[0])
    #print(np.bincount(admeans))
    # print("s", meanseglen, maxseglen, nonzero)
    # print("e", entropy(admeans), entropy(xmeans))
    # print("v", advar, xvar)
    # print("*", entropy(admeans)*advar, entropy(xmeans)*xvar)
    # print("p", pent)
    #return xent*nonzero if maxseglen >= minlen else 0
    #return np.max(summary) #(20, 0.3) 0.5183077935270706 0.5488571088790347
    #return xvar*nonzero if maxseglen >= minlen else 0 #0.5285681381755871 0.5486171025288525
    #return np.max(summary) #(10, 0.4) 0.5118378874621198 0.5431795285615351
    #return xvar*xdiffvar*nonzero if maxseglen >= minlen else 0 #0.528124128808957 0.5435334779559963
    #return xvar*decent if maxseglen >= minlen else 0 #0.5254184248747081 0.5406039283674171
    #return meanseglen #if maxseglen >= minlen else 0 #0.5061975025699631 0.5391262216955992
    #return xent*xvar*nonzero if maxseglen >= minlen else 0 #0.511143729873462 0.5385528126486275
    #return xvar*xdiffent*nonzero if maxseglen >= minlen else 0 #0.5004686802145721 0.5379346988947449
    #return adent*advar*xent*xvar if maxseglen >= minlen else 0 #0.5027575872168802 0.5355081591763832
    #return xvar*nonzero*meanseglen if maxseglen >= minlen else 0 #0.5173722584948066 0.5350565320835667
    #return xvar*nonzero*maxseglen #0.5088370030634288 0.5312888863716688
    #return xdiffvar*nonzero if maxseglen >= minlen else 0 #0.47609386764634976 0.5217101857451909 !!
    #return meanseglen*nonzero*xvar if maxseglen >= minlen else 0
    #return  if maxseglen >= minlen else 0 #6
    #return nonzero/pent if maxseglen >= minlen else 0
    #return decent*xvar if maxseglen >= minlen else 0#xent*xvar
    return nonzero/xent*xdiffent#if mindist > minlen else 0 #*log(len(segs))#/minseglen #if maxseglen >= minlen else 0
    #return nonzero*xvar if maxseglen >= minlen else 0

#def simple_rich_matrix

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