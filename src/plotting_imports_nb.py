import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import matplotlib as mpl

#mpl.rcParams['legend.handlelength'] = 2
#patterns = [[],[5,5],[2,2],[5,5,2,5],[2,2,2,5,5,5],[2,2,2,2,2,5,5,5],[10,5,2,5],[5,5,5,5,2,5],[10,5,2,5,2,5]]
patterns = [[],[3,3],[1,1],[3,3,1,3],[1,1,1,3,3,3],[1,1,1,1,1,3,3,3],[6,3,1,3],[3,3,3,3,1,3],[6,3,1,3,1,3]]
markers = ['o','s','v','p','8','D','>','^']
lcolors = ['k','r','g','y','b','grey','purple','brown']

imw_max = 12
w_im = imw_max/4
h_im = 2.5
sns.set_context("notebook",font_scale=1.15,rc={'lines.linewidth' : 1.5})

sns.set_style('ticks',{'xtick.major.size' : 4,
                       'ytick.major.size' : 4,
                       'axes.spines.top' : False,
                       'axes.spines.right' : False})

from collections import namedtuple

PlotSettings = namedtuple('PlotSettings', ['patterns', 'markers', 'colors', 'w', 'h'])
ps_nb = PlotSettings(patterns=patterns,markers=markers,colors=lcolors,w=w_im,h=h_im)
