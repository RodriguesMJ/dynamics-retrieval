import matplotlib
import scipy.io

matplotlib.use("Agg")  # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams["agg.path.chunksize"] = 100000
import joblib
import matplotlib.pyplot
import numpy


def get_meanden(mode, label):
    folder = "./output_figures/out_mode_0_%d_h5grid_8" % mode
    filename = "%s/meannegden_map_modes_0_%d%s.mat" % (folder, mode, label)
    mat = scipy.io.loadmat(filename)
    meannegden = mat["meannegden"]
    filename = "%s/meanposden_map_modes_0_%d%s.mat" % (folder, mode, label)
    mat = scipy.io.loadmat(filename)
    meanposden = mat["meanposden"]
    return meannegden, meanposden


def plot_site_density_simple():

    folder = "."

    # Atom in 6g7h
    atom = 0  # RET - C20

    filename = "%s/results_C20_6g7k/meannegden_sigcutoff_0.0.jbl" % (folder)
    meannegden = joblib.load(filename)
    filename = "%s/results_C20_6g7k/meanposden_sigcutoff_0.0.jbl" % (folder)
    meanposden = joblib.load(filename)
    print meannegden.shape, meanposden.shape

    times = joblib.load("%s/t_r_p_0.jbl" % folder)
    print times.shape
    idxs = range(0, 104600, 100)
    times_to_plot = times[idxs]
    print times_to_plot.shape

    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.gca().tick_params(axis="x", labelsize=38)
    matplotlib.pyplot.gca().tick_params(axis="y", labelsize=0)
    matplotlib.pyplot.scatter(times_to_plot, meanposden[atom, :], c="b")
    matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c="k")
    matplotlib.pyplot.xlabel("Time [fs]", fontsize=45)
    matplotlib.pyplot.ylabel("Electron density [a.u.]", fontsize=45)
    figname = "%s/results_C20_6g7k/RET_C20.png" % (folder)
    matplotlib.pyplot.savefig(figname)
    matplotlib.pyplot.close()


def plot_site_density():
    label = "_radius_1.7_dist_0.2_sig_4"
    folder = "./output_figures"

    # Atom in 6g7h
    # atom = 1645 # LYS216 - CD
    # atom = 1646 # LYS216 - CE
    # atom = 1647 # LYS216 - NZ

    # atom = 1791 # RET - C7
    # atom = 1792 # RET - C8
    # atom = 1793 # RET - C9
    # atom = 1794 # RET - C10
    # atom = 1795 # RET - C11
    # atom = 1796 # RET - C12
    # atom = 1797 # RET - C13
    # atom = 1798 # RET - C14
    # atom = 1799 # RET - C15
    # atom = 1800 # RET - C16
    # atom = 1801 # RET - C17
    # atom = 1802 # RET - C18
    # atom = 1803 # RET - C19
    atom = 1804  # RET - C20

    modes = range(1, 6)
    for mode in modes:
        print "Mode: ", mode
        meannegden, meanposden = get_meanden(mode, label)

        matplotlib.pyplot.figure(figsize=(40, 10))
        matplotlib.pyplot.gca().tick_params(axis="x", labelsize=25)
        matplotlib.pyplot.gca().tick_params(axis="y", labelsize=0)
        matplotlib.pyplot.scatter(
            range(100, 99200, 100), meannegden[:, atom], c="magenta"
        )
        matplotlib.pyplot.scatter(range(100, 99200, 100), meanposden[:, atom], c="cyan")
        matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c="k")
        matplotlib.pyplot.xlim(100, 99100)
        matplotlib.pyplot.xlabel("Reconstructed frame number", fontsize=28)
        figname = "%s/RET_C20_map_modes_0_%d%s.png" % (folder, mode, label)
        matplotlib.pyplot.savefig(figname)
        matplotlib.pyplot.close()


def plot_peak_density():
    label = "_radius_1_dist_0.1_sig_4_centered_C10"

    mode = 1
    print "Mode: ", mode
    folder = "./out_mode_0_%d_h5grid_8" % mode
    filename = "%s/meanposden_map_modes_0_%d%s.mat" % (folder, mode, label)
    mat = scipy.io.loadmat(filename)
    meanposden = mat["meanposden"]

    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.gca().tick_params(axis="x", labelsize=25)
    matplotlib.pyplot.gca().tick_params(axis="y", labelsize=0)
    matplotlib.pyplot.scatter(range(100, 99200, 100), meanposden[:, 0], c="cyan")
    matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c="k")
    matplotlib.pyplot.xlim(100, 99100)
    matplotlib.pyplot.xlabel("Reconstructed frame number", fontsize=28)
    figname = "%s/C10_peak_map_modes_0_%d%s.png" % (folder, mode, label)
    matplotlib.pyplot.savefig(figname)
    matplotlib.pyplot.close()


def plot_movie_density():
    label = "_radius_1.7_dist_0.2_sig_4"
    modes = range(1, 6)

    x_a = range(29, 214)
    y_a = [-1] * len(x_a)
    x_b = range(239, 451)
    y_b = [-1] * len(x_b)
    x_c = range(576, 761)
    y_c = [-1] * len(x_c)
    x_d = range(774, 947)
    y_d = [-1] * len(x_d)
    x_e = range(956, 1229)
    y_e = [-1] * len(x_e)
    x_f = range(1243, 1473)
    y_f = [-1] * len(x_f)
    x_g = range(1516, 1717)
    y_g = [-1] * len(x_g)
    x_r = range(1785, 1805)
    y_r = [-1] * len(x_r)

    for mode in modes:
        print "Mode: ", mode
        folder = "./out_mode_0_%d_h5grid_8" % mode
        meannegden, meanposden = get_meanden(mode, label)
        nframes = meannegden.shape[0]
        natoms = meannegden.shape[1]
        print nframes
        print natoms

        meannegden_avg = numpy.average(meannegden, axis=0)
        meanposden_avg = numpy.average(meanposden, axis=0)
        print meannegden_avg.shape, meanposden_avg.shape
        matplotlib.pyplot.figure(figsize=(50, 10))
        matplotlib.pyplot.title(
            "Time avg of the integrated positive and negative density surrounding atom sites",
            fontsize=30,
        )
        matplotlib.pyplot.gca().tick_params(axis="x", labelsize=25)
        matplotlib.pyplot.gca().tick_params(axis="y", labelsize=0)
        matplotlib.pyplot.scatter(range(natoms), meannegden_avg, c="magenta")
        matplotlib.pyplot.scatter(range(natoms), meanposden_avg, c="cyan")
        matplotlib.pyplot.scatter(x_a, y_a)
        matplotlib.pyplot.scatter(x_b, y_b)
        matplotlib.pyplot.scatter(x_c, y_c)
        matplotlib.pyplot.scatter(x_d, y_d)
        matplotlib.pyplot.scatter(x_e, y_e)
        matplotlib.pyplot.scatter(x_f, y_f)
        matplotlib.pyplot.scatter(x_g, y_g)
        matplotlib.pyplot.scatter(x_r, y_r)
        matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c="k")
        matplotlib.pyplot.text(
            x_a[0] + 17, y_a[0] + 0.03, "GLU-9 - GLY-31", fontsize=27
        )
        matplotlib.pyplot.text(
            x_b[0] + 27, y_a[0] + 0.03, "ASP-36 - LEU-62", fontsize=27
        )
        matplotlib.pyplot.text(
            x_c[0] + 6, y_a[0] + 0.03, "TRP-80 - VAL-101", fontsize=27
        )
        matplotlib.pyplot.text(
            x_d[0] - 7, y_a[0] + 0.03, "ASP-104 - THR-128", fontsize=27
        )
        matplotlib.pyplot.text(
            x_e[0] + 40, y_a[0] + 0.03, "VAL-130 - GLY-155", fontsize=27
        )
        matplotlib.pyplot.text(
            x_f[0] + 22, y_a[0] + 0.03, "ARG-164 - GLY-192", fontsize=27
        )
        matplotlib.pyplot.text(
            x_g[0] + 4, y_a[0] + 0.03, "PRO-200 - ARG-225", fontsize=27
        )
        matplotlib.pyplot.text(x_r[0] - 8, y_r[0] + 0.03, "RET", fontsize=25)
        matplotlib.pyplot.xlabel("Atom number", fontsize=28)
        figname = "%s/map_modes_0_%d%s_time_avg.png" % (folder, mode, label)
        matplotlib.pyplot.savefig(figname)
        matplotlib.pyplot.close()

        minv = numpy.amin(meannegden)
        maxv = numpy.amax(meanposden)
        print minv, maxv
        for i in range(nframes):
            print mode, i
            matplotlib.pyplot.figure(figsize=(50,10))
            matplotlib.pyplot.title('Integrated positive and negative density surrounding atom sites', fontsize=30)
            matplotlib.pyplot.gca().tick_params(axis='x', labelsize=25)
            matplotlib.pyplot.gca().tick_params(axis='y', labelsize=0)
            matplotlib.pyplot.scatter(range(natoms), meannegden[i,:], c='magenta')
            matplotlib.pyplot.scatter(range(natoms), meanposden[i,:], c='cyan')
            matplotlib.pyplot.scatter(x_a, y_a)
            matplotlib.pyplot.scatter(x_b, y_b)
            matplotlib.pyplot.scatter(x_c, y_c)
            matplotlib.pyplot.scatter(x_d, y_d)
            matplotlib.pyplot.scatter(x_e, y_e)
            matplotlib.pyplot.scatter(x_f, y_f)
            matplotlib.pyplot.scatter(x_g, y_g)
            matplotlib.pyplot.scatter(x_r, y_r)
            matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
            matplotlib.pyplot.ylim(-1.1, maxv+0.1)
            matplotlib.pyplot.text(x_a[0]+17, y_a[0]+0.03, 'GLU-9 - GLY-31',    fontsize=27)
            matplotlib.pyplot.text(x_b[0]+27, y_a[0]+0.03, 'ASP-36 - LEU-62',   fontsize=27)
            matplotlib.pyplot.text(x_c[0]+6,  y_a[0]+0.03, 'TRP-80 - VAL-101',  fontsize=27)
            matplotlib.pyplot.text(x_d[0]-7,  y_a[0]+0.03, 'ASP-104 - THR-128', fontsize=27)
            matplotlib.pyplot.text(x_e[0]+40, y_a[0]+0.03, 'VAL-130 - GLY-155', fontsize=27)
            matplotlib.pyplot.text(x_f[0]+22, y_a[0]+0.03, 'ARG-164 - GLY-192', fontsize=27)
            matplotlib.pyplot.text(x_g[0]+4,  y_a[0]+0.03, 'PRO-200 - ARG-225', fontsize=27)
            matplotlib.pyplot.text(x_r[0]-8,  y_r[0]+0.03, 'RET',               fontsize=25)
            matplotlib.pyplot.xlabel('Atom number', fontsize=28)
            figname = '%s/movie/map_modes_0_%d%s_frame_%0.5d.png'%(folder, mode, label, 100+100*i)
            matplotlib.pyplot.savefig(figname)
            matplotlib.pyplot.close()


def Correlate(x1, x2):
    x1Avg = numpy.average(x1)
    x2Avg = numpy.average(x2)
    numTerm = numpy.multiply(x1 - x1Avg, x2 - x2Avg)
    num = numTerm.sum()
    resX1Sq = numpy.multiply(x1 - x1Avg, x1 - x1Avg)
    resX2Sq = numpy.multiply(x2 - x2Avg, x2 - x2Avg)
    den = numpy.sqrt(numpy.multiply(resX1Sq.sum(), resX2Sq.sum()))
    CC = num / den
    return CC


def plot_helix_density():
    label = "_radius_1.7_dist_0.2_sig_4_time_75000_90000_step_10"
    modes = range(1, 6)  
    for mode in modes:
        print "Mode: ", mode
        meannegden, meanposden = get_meanden(mode, label)
        posden_all = numpy.sum(meanposden, axis=1)
        negden_all = numpy.sum(meannegden, axis=1)
        matplotlib.pyplot.figure(figsize=(10, 20))
        matplotlib.pyplot.scatter(range(1, meanposden.shape[0]+1), posden_all, c='k', s=5)
        matplotlib.pyplot.scatter(range(1, meanposden.shape[0]+1), negden_all, c='k', s=5)
        matplotlib.pyplot.savefig("./allatoms_movie_mode_%d.png" % mode, dpi=4 * 96)
        matplotlib.pyplot.close()


        matplotlib.pyplot.figure(figsize=(20,10))
        color_s = ["blue", 
                   "darkturquoise", 
                   "mediumseagreen", 
                   "magenta", 
                   "black", 
                   "plum", 
                   "purple", 
                   "aquamarine"]
        start_s = [29,  239, 576, 774, 956,  1243, 1516, 1785]
        end_s =   [214, 451, 761, 947, 1229, 1473, 1717, 1805]
        label_s = [
            "Helix A",
            "Helix B",
            "Helix C",
            "Helix D",
            "Helix E",
            "Helix F",
            "Helix G",
            "RET",
        ]
        for i in [1, 2, 5, 6, 7]:
            print i
            color = color_s[i]
            start = start_s[i]
            end = end_s[i]
            print start, end
            h_pos = meanposden[:,start:end]
            h_neg = meannegden[:,start:end]
            h_pos = numpy.sum(h_pos, axis=1)
            h_neg = numpy.sum(h_neg, axis=1)

            matplotlib.pyplot.scatter(range(1, h_pos.shape[0]+1), h_pos, c=color, s=5, label=label_s[i])
            matplotlib.pyplot.scatter(range(1, h_pos.shape[0]+1), h_neg, c=color, s=5)
            matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')

        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig('./helix_movie_mode_%d_simple.png'%mode, dpi=4*96)
        matplotlib.pyplot.close()


def calc_helix_CC():
    label = "_radius_1.7_dist_0.2_sig_4"
    modes = range(1, 6)
    n_helices = 5  # RET, G, C, B, F
    n_modes = 5
    n_times = 991
    posden_helixmeans = numpy.zeros((n_times, n_helices * n_modes))
    negden_helixmeans = numpy.zeros((n_times, n_helices * n_modes))
    idx = 0
    for mode in modes:
        print "Mode: ", mode
        meannegden, meanposden = get_meanden(mode, label)

        start_s = [29, 239, 576, 774, 956, 1243, 1516, 1785]
        end_s = [214, 451, 761, 947, 1229, 1473, 1717, 1805]
        label_s = [
            "Helix A",
            "Helix B",
            "Helix C",
            "Helix D",
            "Helix E",
            "Helix F",
            "Helix G",
            "RET",
        ]
        for i in [7, 6, 2, 1, 5]:
            start = start_s[i]
            end = end_s[i]
            print "Mode: ", mode, label_s[i], start, end
            h_pos = meanposden[:, start:end]
            h_neg = meannegden[:, start:end]
            h_pos = numpy.sum(h_pos, axis=1)
            h_neg = numpy.sum(h_neg, axis=1)
            print idx
            posden_helixmeans[:, idx] = h_pos
            negden_helixmeans[:, idx] = h_neg
            idx = idx + 1

    pos_CC = numpy.zeros((n_helices * n_modes, n_helices * n_modes))
    neg_CC = numpy.zeros((n_helices * n_modes, n_helices * n_modes))
    for j in range(n_helices * n_modes):
        print j
        for k in range(j + 1):
            x1 = posden_helixmeans[:, j]
            x2 = posden_helixmeans[:, k]
            CC = Correlate(x1, x2)
            pos_CC[j, k] = CC
            pos_CC[k, j] = CC

            x1 = negden_helixmeans[:, j]
            x2 = negden_helixmeans[:, k]
            CC = Correlate(x1, x2)
            neg_CC[j, k] = CC
            neg_CC[k, j] = CC

    XTicksLabels = [
        "1-RET",
        "1-G",
        "1-C",
        "1-B",
        "1-F",
        "2-RET",
        "2-G",
        "2-C",
        "2-B",
        "2-F",
        "3-RET",
        "3-G",
        "3-C",
        "3-B",
        "3-F",
        "4-RET",
        "4-G",
        "4-C",
        "4-B",
        "4-F",
        "5-RET",
        "5-G",
        "5-C",
        "5-B",
        "5-F",
    ]

    matplotlib.pyplot.imshow(pos_CC, cmap="cool")
    matplotlib.pyplot.colorbar()
    ax = matplotlib.pyplot.gca()
    ax.xaxis.set(ticks=numpy.arange(0, len(XTicksLabels)), ticklabels=XTicksLabels)
    ax.set_xticklabels(XTicksLabels, rotation=90, minor=False)
    ax.yaxis.set(ticks=numpy.arange(0, len(XTicksLabels)), ticklabels=XTicksLabels)
    ax.set_yticklabels(XTicksLabels, rotation=0, minor=False)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig("CC_posden_RET_G_C_B_F.png")
    matplotlib.pyplot.close()

    matplotlib.pyplot.imshow(neg_CC, cmap="cool")
    matplotlib.pyplot.colorbar()
    ax = matplotlib.pyplot.gca()
    ax.xaxis.set(ticks=numpy.arange(0, len(XTicksLabels)), ticklabels=XTicksLabels)
    ax.set_xticklabels(XTicksLabels, rotation=90, minor=False)
    ax.yaxis.set(ticks=numpy.arange(0, len(XTicksLabels)), ticklabels=XTicksLabels)
    ax.set_yticklabels(XTicksLabels, rotation=0, minor=False)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig("CC_negden_RET_G_C_B_F.png")
    matplotlib.pyplot.close()


def correlate_density():
    matplotlib.pyplot.rcParams["legend.loc"] = "center right"
    label = "_radius_1.7_dist_0.2_sig_4"
    modes = range(1, 6)
    results_path = "../../../bR_light_q_16384_b_5000"
    VT_final = joblib.load("%s/VT_final.jbl" % (results_path))

    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    print "nmodes: ", nmodes
    print "s: ", s

    for mode in modes:
        print "Mode: ", mode
        meannegden, meanposden = get_meanden(mode, label)
        posden_all = numpy.sum(meanposden, axis=1)
        negden_all = numpy.sum(meannegden, axis=1)
        chrono = VT_final[mode, :]
        matplotlib.pyplot.figure(figsize=(20, 10))
        matplotlib.pyplot.scatter(
            range(100, 100 * (meanposden.shape[0] + 1), 100),
            posden_all,
            c="b",
            s=5,
            label="",
            zorder=100,
        )
        matplotlib.pyplot.scatter(
            range(100, 100 * (meanposden.shape[0] + 1), 100), negden_all, c="b", s=5
        )
        matplotlib.pyplot.scatter(
            range(chrono.shape[0]),
            abs(chrono),
            c="darkturquoise",
            s=5,
            label="",
            zorder=200,
        )
        matplotlib.pyplot.scatter(
            range(chrono.shape[0]),
            chrono ** 2,
            c="navajowhite",
            s=5,
            label="",
            zorder=0,
        )
        matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c="k")
        matplotlib.pyplot.legend(fontsize=16)
        matplotlib.pyplot.savefig(
            "./allatoms_movie_mode_%d_chrono.png" % mode, dpi=4 * 96
        )
        matplotlib.pyplot.close()

        abs_chrono = abs(chrono[range(100, 100 * (meanposden.shape[0] + 1), 100)])
        chrono_sq = (chrono[range(100, 100 * (meanposden.shape[0] + 1), 100)]) ** 2
        CC_posden_abs = Correlate(posden_all, abs_chrono)
        print CC_posden_abs
        CC_posden_sq = Correlate(posden_all, chrono_sq)
        print CC_posden_sq
        CC_negden_abs = Correlate(negden_all, abs_chrono)
        print CC_negden_abs
        CC_negden_sq = Correlate(negden_all, chrono_sq)
        print CC_negden_sq


if __name__ == "__main__":
    plot_site_density_simple()
    # plot_movie_density()
    # correlate_density()
    # calc_helix_CC()
