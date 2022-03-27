import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.io  
import os
import json
#import itertools
from re import L

#from scipy.stats import percentileofscore
from loess.loess_1d import loess_1d
from functools import reduce
from math import floor,ceil
import seaborn as sns

import matplotlib.pyplot as plt 

import matplotlib.transforms as transforms

import matplotlib.path as mpath
from matplotlib.ticker import NullFormatter

from matplotlib import rc
from matplotlib.patches import Ellipse
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib import patches

from itertools import chain


MY_DPI = 96


#
#   Folder Paths
#
HOFASM_RESULTS = "../data/HOFASM_Experiments/"
TAME_RESULTS = "../data/TAME_Experiments/"
TKP_RESULTS = "../data/TKP_Experiments/"

#
#   Color Schemes 
#

color_scheme = 3

simax_blue = "#0a4c88"

if color_scheme == 1:
    red_c = '#fb6b5b' # Red
    red_dark_c = "#E34635"
    red_darkest_c =  "#BB2617"

    green_c = "#00a878"
    green_light_c = "#A7EBD8"
    green_dark_c = "#00664A"
    green_c_shift = green_c 

    purple_c ="#aa42a3"
    purple_dark_c = "#7E1275"

    blue_c = '#7ecbe0'
    blue_dark_c = simax_blue
    darkest_blue_c = simax_blue

    black_c = "k"

    orange_c = "orange"

    yellow_c = "#fbc04c"
    yellow_darkest_c = "#EAA622"

    pink_c = "#D65780"

elif color_scheme == 2:
    red_c = "#D64550"
    #color2 = "#1B998B"  
    green_c = "#06BA63"
    green_c_shift = green_c 


    green_dark_c ="#31572C" 
    purple_c = "#8C2155" # Purple 
    blue_c = "#446DF6" # Blue
    blue_dark_c = "#0E3D95" # Dark Blue
    yellow_c = "#E6AF2E" # Yellow
    orange_c = "#FA8334" # Orange  
    black_c = "#34252F"
    brown_c = "#3B0D11"
    pink_c = "#D65780" #unused
elif color_scheme == 3:
    red_c = "#E83521" # Red
    green_dark_c = "#235921"
    green_c_old = "#68C665"
    green_c ="#489C53" #"#68C665" #alt color:"#05B393"
    green_shift_c = "#68C665" 
                    # use a different color for the shift color in fig 5.1a)
    green_light_c = "#5CCABD"
    purple_c = "#4B257D"
    blue_c = "#5045F5"
    blue_dark_c = "#2837A8"
    yellow_c = green_light_c#"#DEA908"
    orange_c = "#F08A3F"
    black_c = "#2A1A1F"
    brown_c = "#B4512D"
    pink_c = "#D65780"
else:
    pass

checkboard_color = [.95]*3

tick_length = 5

#
#  Algorithm Colors
#

T_color = red_c
T_linestyle = "solid"
T_marker = 's'


LT_color = green_c
LT_linestyle = ":"#(0, (5, 5)) #"--"
LT_marker='^'

LT_Klau_color = orange_c
LT_Klau_linestyle = "-."#(0, (2.5, 2.5))# (0,(3,1,1,1,1,1))
LT_Klau_marker = LT_marker

LT_Tabu_color = blue_dark_c
LT_Tabu_linestyle = "solid"#(0, (7.5, 7.5))#(0,(10,1.5))
LT_Tabu_marker = None

# Update 
LT_rom_color = green_dark_c
LT_rom_linestyle="solid"#(0,(10,1.5))
LT_rom_marker=LT_marker 
        # facecolor=none



#  
#    Marker Augmentations
#
Klau_marker_aug = "+"
def make_LT_Klau_marker(marker): 
    p1 = MarkerStyle(marker).get_path()
    p2 = MarkerStyle(Klau_marker_aug).get_path()

    new_verts = np.concatenate([
        p1.vertices,
        [[np.nan, np.nan]],
        p2.vertices,
        [[np.nan, np.nan]],
    ])
    new_codes = np.concatenate([
        p1.codes,
        [Path.MOVETO],
        p2.codes,
        [Path.CLOSEPOLY]
    ])

    return MarkerStyle(Path(new_verts,new_codes))



#TODO: refactor this to LREA_[trait]
LREigenAlign_color = brown_c
LREigenAlign_linestyle = ":"#(0,(1.5,3))
LREigenAlign_marker = "*"

LREA_Tabu_color = green_dark_c #black_c
LREA_Tabu_linestyle = "solid"##(0,(1.5,5))
LREA_Tabu_marker = None

LREA_Klau_color = purple_c
LREA_Klau_linestyle= "-."#(0,(1.5,1))
LREA_Klau_marker = None

LGRAAL_color = black_c
LGRAAL_linestyle = ":"#(0,(3,1,1,1,1,1))
LGRAAL_marker = 'x'

LRT_color = blue_c 
LRT_linestyle = ":"#(0, (5, 2, 2, 2))#(0, (3, 1, 1, 1))
LRT_marker = "o"

LRT_Tabu_color = yellow_c
LRT_Tabu_linestyle = "solid"#(0, (10, 3, 3, 3))#"solid"
LRT_Tabu_marker = None

LRT_Klau_color = pink_c
LRT_Klau_linestyle = "-."#"solid"#(0, (2.5, 1, 1, 1))#":"
LRT_Klau_marker = None


LRT_lrm_color = blue_dark_c 
LRT_lrm_linestyle = "solid"#"-."
LRT_lrm_marker = LRT_marker
           # facecolor=none

LRT_lrm_Klau_color = 'r'


def show_colors_linestyles():

    f = plt.figure() 
    ax = plt.gca()

    styles_to_plot = [
        ("TAME",T_color,T_linestyle,T_marker),
        (r"$\Lambda$-TAME",LT_color,LT_linestyle,LT_marker),
        (r"$\Lambda$T-Klau",LT_Klau_color,LT_Klau_linestyle,LT_Klau_marker),
        (r"$\Lambda$T-LocalSearch",LT_Tabu_color,LT_Tabu_linestyle,LT_Tabu_marker),
        ("LowRankTAME",LRT_color,LRT_linestyle,LRT_marker),
        ("LRT-Klau",LRT_Klau_color,LRT_Klau_linestyle,LRT_Klau_marker),
        ("LRT-LS",LRT_Tabu_color,LRT_Tabu_linestyle,LRT_Tabu_marker),
        ("LowRankEigenAlign",LREigenAlign_color,LREigenAlign_linestyle,LREigenAlign_marker),
        ("LREA-Klau",LREA_Klau_color,LREA_Klau_linestyle,LREA_Klau_marker),
        ("LREA-LS",LREA_Tabu_color,LREA_Tabu_linestyle,LREA_Tabu_marker),
        ("LGRAAL",LGRAAL_color,LGRAAL_linestyle,LGRAAL_marker),
    ]

    n = 10 
    total_styles = len(styles_to_plot)
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
    for ((label,color,linestyle, marker),y) in zip(styles_to_plot,np.linspace(.95,.05,len(styles_to_plot))):
        ax.plot(np.linspace(0,1,n),[y]*n,linestyle=linestyle,marker=marker,c=color)
        ax.annotate(label,xy=(.5,y),xycoords="data",c=color,va="center",ha="center").set_bbox(bbox)

    plt.show()


def test_outline(marker=LT_marker): 

    f = plt.figure()

    p1 = MarkerStyle(marker).get_path()
    print(p1)
    p2 = MarkerStyle(marker,fillstyle ='none').get_path()
    print(p2)
    new_verts = np.concatenate([
        #p1.vertices,
        1.2*p2.vertices,
    ])
    new_codes = np.concatenate([
        #p1.codes,
        p2.codes,
    ])

    return MarkerStyle(Path(new_verts,new_codes))

#
#  Custom Patches
#

plus_patch_path = [[.25, 0], [.5, 0], [.5, .25],[.75, .25],[.75, .5],[.5, .5],[.5, .75],[.25, .75],[.25, .5],[0, .5],[0, .25],[.25,.25]]
#plus_patch = patches.Polygon(plus_patch_path)



