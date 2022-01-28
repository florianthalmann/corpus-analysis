
#moved here from salami.py
def compare_lsd_snf_ssm():
    def getCSMCosine(X, Y):
        XNorm = np.sqrt(np.sum(X**2, 1))
        XNorm[XNorm == 0] = 1
        YNorm = np.sqrt(np.sum(Y**2, 1))
        YNorm[YNorm == 0] = 1
        D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
        D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
        return D
    
    hop_length = 512
    wins_per_block = 20
    win_fac=10
    y, sr = librosa.load(get_audio(749), sr=22050)
    C = np.abs(librosa.cqt(y=y, sr=sr))
    _, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)#, start_bpm=240)
    bintervals = librosa.util.fix_frames(beats, x_max=C.shape[1])
    nHops = int((y.size-hop_length*win_fac*wins_per_block)/hop_length)
    intervals = np.arange(0, nHops, win_fac)
    
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    chroma = librosa.util.sync(chroma, intervals, aggregate=np.median)
    print(chroma.shape, len(intervals), librosa.util.sync(chroma, bintervals, aggregate=np.median).shape, len(bintervals))
    n_frames = chroma.shape[1]
    chroma = chroma[:, :n_frames]
    XChroma = librosa.feature.stack_memory(chroma, n_steps=1, mode='edge').T
    DChroma = 1-getCSMCosine(XChroma, XChroma)
    
    plot_matrix(DChroma, PLOT_PATH+"749-<.png")
    
    print(DChroma.shape, len(bintervals))
    DChroma = librosa.util.sync(DChroma, bintervals, aggregate=np.mean)
    print(DChroma.shape)
    DChroma = librosa.util.sync(DChroma.T, bintervals, aggregate=np.mean)
    print(DChroma.shape)
    #DChroma = 1-getCSMCosine(XChroma, XChroma)
    plot_matrix(DChroma, PLOT_PATH+"749-<<.png")
    
    BPO = 12 * 3
    N_OCTAVES = 7
    REC_WIDTH = 9
    
    yh = librosa.effects.harmonic(y, margin=8)
    
    #print('\tcomputing CQT...')
    C = librosa.amplitude_to_db(librosa.cqt(y=yh, sr=sr,
                                            bins_per_octave=BPO,
                                            n_bins=N_OCTAVES * BPO),
                                ref=np.max)
    
    #print('\ttracking beats...')
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    
    R = librosa.segment.recurrence_matrix(Csync, width=REC_WIDTH,
                                          mode='affinity',
                                          metric='cosine',
                                          sym=True)
    
    plot_matrix(R, PLOT_PATH+"749-...png")
    print(DChroma.shape, R.shape)