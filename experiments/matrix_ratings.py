import math
import numpy as np
from corpus_analysis.alignment.affinity import to_diagonals, matrix_to_segments,\
    segments_to_matrix
from corpus_analysis.util import plot_matrix, flatten, summarize_matrix
from corpus_analysis.stats.util import entropy, entropy2

def matrix_rating_s(matrix, resolution=10, minlen=9):
    #np.fill_diagonal(matrix, 0)
    if np.sum(matrix) == 0: return 0
    matrix = segments_to_matrix([s for s in matrix_to_segments(matrix) if len(s) > 2], matrix.shape)
    #diagonals = to_diagonals(matrix)
    #antidiagonals = to_diagonals(np.flip(matrix, axis=0))
    xmeans, xvar, xent = distribution_measures(matrix, resolution)
    #dmeans, dvar, dent = distribution_measures(diagonals, resolution)
    #admeans, advar, adent = distribution_measures(antidiagonals, resolution)
    nonzero = len([x for x in xmeans if 0 < x < resolution-1]) / len(matrix)
    rnonzero = len([x for x in matrix if 0 < np.sum(x)]) / len(matrix)
    #print(np.min([np.sum(x) for x in matrix]), np.max([np.sum(x) for x in matrix]))
    #nonzero = len([x for x in matrix if np.sum(x) > 0]) / len(matrix)
    #decent = len([x for x in xmeans if 2 < np.sum(x) < 0.2*len(xmeans)]) / len(xmeans)
    decent = len([x for x in matrix if 0 < np.sum(x) < 0.1*len(matrix)]) / len(matrix)
    segs = [len(s) for s in matrix_to_segments(matrix)]
    #ones = (len([s for s in segs if s < round(matrix.shape[0]/50)])+1)/len(segs)
    ones = (len([s for s in segs if s == 1])+1)/len(segs)
    short = (len([s for s in segs if s < 4])+1)/len(segs)
    #segs = [s for s in segs if 4 < s]# < 20]
    #segs = [s for s in segs if 0.02*len(matrix) < s]# < 20]
    segs = [0] if len(segs) == 0 else segs
    meanseglen, maxseglen, minseglen = np.mean(segs), np.max(segs), np.min(segs)
    #pent = entropy(matrix_to_endpoint_vector(matrix))
    #pmeans, pvar, pent = distribution_measures(matrix_to_endpoint_vector(matrix), resolution)
    #mindist = min_dist_between_nonzero(dmeans)
    #xdiff = np.abs(np.diff([np.sum(x) for x in matrix]))
    xdiff = np.abs(np.diff(xmeans))
    xdiffent = entropy(xdiff)
    xdiffvar = np.std(xdiff)/np.mean(xdiff)
    # addiff = np.abs(np.diff(admeans))
    # addiffent = entropy(addiff)
    # addiffvar = np.std(addiff)/np.mean(addiff)
    #windows = np.concatenate(strided2D(matrix, 5))
    #print(len(windows), windows[0])
    # summary = summarize_matrix(matrix.astype(int), round(len(matrix)/50))
    # zeroblocks = len(summary[summary<=1])/(len(matrix)**2)
    # print(zeroblocks)
    # zeroblocks = 0.5-abs(0.5-zeroblocks)
    # print(zeroblocks)
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
    #return nonzero/ones*xvar
    #maxseglen = math.log(maxseglen)
    totalmat = len(np.nonzero(matrix)[0])
    totalsegs = np.sum(segs)
    return rnonzero*math.log(meanseglen)/xvar#maxseglen*xvar*decent/xdiffent#*xent/xdiffent #if rnonzero >= 0.3 and len(segs) > 10 else 0 #*math.log(len(segs))/ones
    #return  if maxseglen >= minlen else 0 #6
    #return nonzero/pent if maxseglen >= minlen else 0
    #return decent*xvar if maxseglen >= minlen else 0#xent*xvar
    #return nonzero/xent*xdiffent#if mindist > minlen else 0 #*log(len(segs))#/minseglen #if maxseglen >= minlen else 0
    #                   mat         smat
    #nonzero            .386 .440   .392 .442
    #xvar               .460 .472   .468 .482
    #1/ones             .457 .467   .407 .451
    #nonzero/ones*xvar  .461 .472   .400 .445
    #1/xent             .448 .464   .464 .477
    #xvar/xent          .452 .467   .470 .481
    
    
    #crema full                     (.5075)   smat
    #rnonzero*log(meanseglen)       .5043     .5066     #less good: rnonzero*log(meanseglen)/xvar
    #nonzero                        .5054     .5054
    #rnonzero*meanseglen            .5022     .5051
    #xent                           .5015     .5043
    #totalsegs/totalmat             .5007     .5028
    #xdiffvar                       .4998     .4963
    #xvar                           .4955     .4932
    
    
    #crema base         (.546 .560)
    #totalsegs/totalmat .538 .550
    #xdiffvar           .532 .554 *rnonzero same
    #xent               .525 .553
    #xvar               .512 .537
    
    #smat
    #rnonzero*xent      .530 .560
    
    
    
    #mat
    #rnonzero*xent/xdiffent .499 .525
    #xent/xdiffent          .497 .521
    #rnonzero*xent          .484 .519
    #rnonzero*xvar          .491 .518
    #rnonzero*xdiffvar      .476 .518
    #xdiffvar               .478 .515
    #xent                   .477 .511
    #xdiffvar/xdiffent      .471 .506
    #xvar/xdiffent          .484 .504
    #xvar                   .477 .499
    #1/xdiffent             .472 .496
    
    #mat25-w4
    #maxseglen*xvar*decent/xdiffent     .5115
    
    #mat25-w
    #meanseglen*rnonzero    .502
    #rnonzero*xvar          .497
    #meanseglen             .496
    #xvar                   .4917
    #rnonzero*xdiffvar*xvar .4915
    #rnonzero*xent          .488
    #xent/xdiffent          .487
    #xdiffvar               .471
    #rnonzero*xent/xdiffent .470
    #rnonzero*xdiffvar      .468
    #1/pent                 .446
    
    #smat25-w
    #meanseglen             .483    .512
    #rnonzero*xdiffvar      .4868   .510
    #1/pent                 .471    .502
    #xdiffvar               .481    .494
    #rnonzero*xdiffvar*xvar .4873   .493
    #mindist                .464    
    #rnonzero*xvar                  .480
    #xvar                   .459    .478
    #1/xdiffent             .452
    #rnonzero*xent                  .448
    #1/xent                 .444
    
    #mat
    #xvar                       .474 .489 (if rnonzero >= 0.3 else 0) segs > 1
    #xvar                       .472 .485 (if rnonzero >= 0.3 else 0)
    #xvar*xdiffvar              .471 .483 (if rnonzero >= 0.3 else 0)
    #xvar*xdiffvar*nonzero      .465 .477 (real xdiffvar!)
    #xvar*xdiffvar*nonzero      .463 .475
    #nonzero/ones*xvar          .461 .472
    #xvar                       .460 .472
    #meanseglen*nonzero*xvar    .462 .471 (2 < s < 30)
    #xvar/ones                  .461 .471
    #1/ones                     .457 .467
    
    #smat
    #xvar               .468 .482
    #xvar/xent          .470 .481
    #xvar*advar         .467 .478
    #1/xent             .464 .477
    #advar              .466 .473

def matrix_rating_b(matrix, resolution=10, minlen=10):
    np.fill_diagonal(matrix, 0)#remove the main diagonal
    if np.sum(matrix) == 0: return 0
    diagonals = to_diagonals(matrix)
    antidiagonals = to_diagonals(np.flip(matrix, axis=0))
    xmeans, xvar, xent = distribution_measures(matrix, resolution)
    dmeans, dvar, dent = distribution_measures(diagonals, resolution)
    admeans, advar, adent = distribution_measures(antidiagonals, resolution)
    nonzero = len([x for x in xmeans if np.sum(x) > 0]) / len(xmeans)
    rnonzero = len([x for x in matrix if 0 < np.sum(x)]) / len(matrix)
    decent = len([x for x in matrix if 2 < np.sum(x) < 0.1*len(matrix)]) / len(xmeans)
    segs = [len(s) for s in matrix_to_segments(matrix)]
    # segs = [s for s in segs if s > 4]
    # segs = [0] if len(segs) == 0 else segs
    meanseglen, maxseglen, minseglen = np.mean(segs), np.max(segs), np.min(segs)
    medianseglen = np.median(segs)
    # pent = entropy(matrix_to_endpoint_vector(matrix))
    # mindist = min_dist_between_nonzero(dmeans)
    xdiff = np.abs(np.diff(xmeans))
    xdiffent = entropy(xdiff)
    dists = np.diff(np.where(np.hstack(matrix))[0])
    meandist, mindist = np.mean(dists), np.min(dists)
    distent = entropy(dists)
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
    #return nonzero/xent*xdiffent#if mindist > minlen else 0 #*log(len(segs))#/minseglen #if maxseglen >= minlen else 0
    #print(decent, mindist, meandist)
    numsegs = len(segs)
    return rnonzero*math.log(maxseglen)/xvar#decent*distent/xent#*math.log(mindist)*math.log(meandist)#*(meanseglen/len(matrix))*xvar#rnonzero/adent*xvar*dent #if rnonzero > 0.5 else 0
    #return nonzero*xvar if maxseglen >= minlen else 0
    
    #crema full             (.5075)
    #rnonzero*log(maxseglen)/xvar  .50868
    #rnonzero*log(maxseglen).5092   .50858
    #rnonzero*maxseglen     .5090   .50857
        #rnonzero*maxseglen/xvar.5089   .50845
    #rnonzero               .5087
        #rnonzero*maxseglen*log(numsegs) .5085
    #maxseglen*log(numsegs) .5082
        #rnonzero*log(numsegs)  .5079
        #rnonzero/xvar          .5077
    #maxseglen              .5075   .5056
        #rnonzero/advar         .5075
    #numsegs                .5072
        #rnonzero/dvar          .5072
        #rnonzero/xent/xvar     .5063
        #rnonzero/xent          .5058
        #rnonzero*meanseglen    .5045
    #xent/xvar              .5035
    #meanseglen             .5033   .5035
        #rnonzero*xdiffent      .5030
    #xdiffent               .5009
    #xent                   .4995
    #medianseglen           .4995
    #minseglen              .4937
    #xvar                   .4933
    #advar                  .4931
    #dvar                   .4902
    
    
    #maxseglen*log(numsegs)     .5082
    #maxseglen*numsegs          .5080
    #log(maxseglen)*log(numsegs).5078
    #log(maxseglen)*numsegs     .5074
    
    
    #crema                  (.546)
    #rnonzero*xent/xvar     .551
    #xent/xvar              .548
    #rnonzero/xvar          .544
    #rnonzero*xdiffent      .543
    #rnonzero*xent          .538
    #xdiffent               .537
    #rnonzero*meanseglen    .534
    #meanseglen             .531
    #rnonzero               .531
    #xent                   .524
    #xvar                   .509
    
    
    
    #nonzero:   .519 .479   .462 .450
    #xent:      .505 .502   .456 .455
    #xdiffent:  .498 .506   .456 .457
    #xvar:      .481 .509   .453 .461
    #n/xv:      .512 .485   .461 .453
    #n/xde:     .512 .489   .455 .455

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