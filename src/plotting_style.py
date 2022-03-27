from dataProcessing import * 

#src: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

#src: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i != j:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts

def make_runtime_heatmaps():
    graph_names, LGRAAL_tri_results, LGRAAL_runtimes, Original_TAME_tri_results, Original_TAME_runtimes, new_TAME_runtimes, new_TAME_tri_results = get_results([6, 5, 1, 9, 4, 3, 8, 0, 7, 2])

    n = len(graph_names)

    data = np.zeros((n,n))

    for i,j in zip(*np.triu_indices(n,k=1)):
       #data[i,j] = Original_TAME_runtimes[i,j]/new_TAME_runtimes[i,j]
        data[i,j] = np.log10(Original_TAME_runtimes[i,j]/new_TAME_runtimes[i,j])

    for i, j in zip(*np.tril_indices(n, k=-1)):
       #data[i, j] = LGRAAL_runtimes[i, j] / new_TAME_runtimes[i, j]
        data[i, j] = np.log10(LGRAAL_runtimes[i, j] / new_TAME_runtimes[i, j])

    datam = np.ma.masked_where(data == 0.0,data)

    im,cbar = heatmap(datam, graph_names, graph_names, ax=None,cmap = gradient_cmap(lightest_t2_color,darkest_t2_color),
                cbar_kw={}, cbarlabel="$\log_{10}$ runtime improvement")
    annotate_heatmap(im, valfmt="{x:.1f}")
    plt.show()

#assuming hex input
def gradient_cmap(c1,c2):

    c1_rbga = mpl.colors.to_rgba_array(c1)[0]
    c2_rbga = mpl.colors.to_rgba_array(c2)[0]

   #return c1_rbga,c2_rbga
    N1 = 256
    N2 = 0

    vals = np.ones((N1 + N2, 4))
    vals[:N1, 0] = np.linspace(c1_rbga[0], c2_rbga[0], N1)
    vals[:N1, 1] = np.linspace(c1_rbga[1], c2_rbga[1], N1)
    vals[:N1, 2] = np.linspace(c1_rbga[2], c2_rbga[2], N1)

    """
    vals[N1:, 0] = np.linspace(1,c2_rbga[0],  N2)
    vals[N1:, 1] = np.linspace(1,c2_rbga[1],  N2)
    vals[N1:, 2] = np.linspace(1,c2_rbga[2],  N2)
    """
    return ListedColormap(vals)

def make_circle(ax,x,y):
    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    width = .15 * maxd / dx
    height = .15 * maxd / dy

    return Ellipse((x, y), width, height)



def plot_percentiles(ax, data, x_domain, lines, ribbons,**kwargs):

    """------------------------------------------------------------------------
        Plots lines and ribbons on a given mpl axis ax. 
      
    Inputs:
    -------
    ax - (mpl axis)
    data - (2D np.array of N)
        lists are all expected to be the same length.
    x_domain - ()
    lines - ({1D np.array -> list of N, Float,String})
        Lines are functions applied to each element of every list, thus plots
        l(i). Passed in a tuple with alpha and color specifications. 
    ribons - ({Float,Float,Float,colors})
        First two floats represent upper and lower percentiles to bound with 
        ribbon, followed by alpha and color string. 
    -----------------------------------------------------------------------"""
    n,m = data.shape

    #reshaped_data = data
    #reshaped_data = np.reshape(data, (n * m, iters))
    for (lower_percentile,upper_percentile,alpha,color) in ribbons:
        ax.fill_between(x_domain,
                        np.percentile(data, lower_percentile, axis=0),
                        np.percentile(data, upper_percentile, axis=0),
                        facecolor=color,alpha=alpha)


    for (col_func,alpha,color) in lines:
        line_data = []

        for j in range(m):
            line_data.append(col_func(data[:,j]))


        ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

from matplotlib.colors import ListedColormap, BoundaryNorm
def plot_percentiles_thresholded(ax, data, x_domain,threshold,above_color,below_color, lines, ribbons,**kwargs):
    """------------------------------------------------------------------------
        Plots lines and ribbons with split colors determined by threshold on a 
      given mpl axis.
      
    Inputs:
    -------
    ax - (mpl axis)
    data - (2D np.array of N)
        lists are all expected to be the same length.
    x_domain - ()
    threshold - (float)
        The point on the y axis to split the colors. 
    lines - ({1D np.array -> list of N, Float,String})
        Lines are functions applied to each element of every list, thus plots
        l(i). Passed in a tuple with alpha and color specifications. 
    color{1,2} - (#TODO: get color types from pyplot)
        the colors of the split. 
    ribons - ({Float,Float,Float,colors})
        First two floats represent upper and lower percentiles to bound with 
        ribbon, followed by alpha and color string. 
    -----------------------------------------------------------------------"""

    n,m = data.shape

    #cmap = ListedColormap([color1, color2])
    #norm = BoundaryNorm([np.min(data), threshold, np.max(data)], cmap.N)    

    for (lower_percentile,upper_percentile,alpha) in ribbons:

        lower_bound = np.percentile(data, lower_percentile, axis=0)
        upper_bound = np.percentile(data, upper_percentile, axis=0)
        #if np.min(np.percentile(data, lower_percentile, axis=0)) >= threshold:
        #    ax.fill_between(x_domain,lower_bound,upper_bound,alpha=alpha,c=color1)
        """
        ax.fill_between(x_domain,lower_bound,upper_bound,
                        where=lower_bound >= threshold,interpolate=True,
                        alpha=alpha,facecolor=color1)
        """
        ax.fill_between(x_domain,lower_bound,[threshold]*data.shape[1],
                        interpolate=True,#where=lower_bound <= threshold,
                        alpha=alpha,facecolor=below_color)
        ax.fill_between(x_domain,[threshold]*data.shape[1],upper_bound,
                        interpolate=True,#where=lower_bound <= threshold,
                        alpha=alpha,facecolor=above_color)

        #white out where opposite bounds dip above or below threshold
        ax.fill_between(x_domain,upper_bound,[threshold]*data.shape[1],
                        where=upper_bound < threshold,interpolate=True,
                        alpha=1.0,facecolor="w") 
        ax.fill_between(x_domain,[threshold]*data.shape[1],lower_bound,
                        where=lower_bound >= threshold,interpolate=True,
                        alpha=1.0,facecolor="w")


    for (col_func,alpha,linestyle) in lines:
        line_data = []

        for j in range(m):
            line_data.append(col_func(data[:,j]))
        above_data = np.ma.masked_where(np.array(line_data) < threshold, np.array(line_data))
        below_data = np.ma.masked_where(np.array(line_data) > threshold, np.array(line_data))
        ax.plot(x_domain,above_data,alpha=alpha,c=above_color,linestyle=linestyle,**kwargs)
        ax.plot(x_domain,below_data,alpha=alpha,c=below_color,linestyle=linestyle,**kwargs)

def plot_1d_loess_smoothing(domain,data,frac,ax=None,logFilter=False,logFilterAx="both",**kwargs):
    """ 
    ax - (mpl axis)
    domain - (list of N)
        - expected ot be convertable to np.array
    data - ()
    logFilter - running on data intended for a log plot tends to fail 
                when the scales are large. converting the x and y data
                to log10 and then back seems to address the issue.
    """
    if logFilter:
        if logFilterAx == "both" or logFilterAx == "x":
            domain = np.log10(domain)
        if logFilterAx == "both" or logFilterAx == "y":
            data = np.log10(data)
    
    xout, yout, weights = loess_1d(np.array(domain), np.array(data), frac=frac)
    p = sorted(range(len(xout)),key=lambda i: xout[i])
    xout = xout[p]
    yout = yout[p]


    
    if logFilter:

        if logFilterAx == "both" or logFilterAx == "x":
            xout = [10.0**x for x in xout]

        if logFilterAx == "both" or logFilterAx == "y":
            yout = [10.0**y for y in yout]
 

    if ax is None:
        return xout, yout
    else:
        ax.plot(xout,yout,**kwargs)

def plot_1d_loess_smoothing_across_samples(domain,data,frac,ax=None,logFilter=False,logFilterAx="both",**kwargs):
    """ 
    ax - (mpl axis)
    domain - (list of N)
        - expected ot be convertable to np.array
    data - ()
    logFilter - running on data intended for a log plot tends to fail 
                when the scales are large. converting the x and y data
                to log10 and then back seems to address the issue.
    """
    
    flattened_domain = []
    flattened_data = []
    for datum in data:
        for i,val in enumerate(datum):
            flattened_data.append(val)
            flattened_domain.append(domain[i] + 1e-6*np.random.rand())

    if logFilter:
        if logFilterAx == "both" or logFilterAx == "x":
            flattened_domain = np.log10(flattened_domain)
        if logFilterAx == "both" or logFilterAx == "y":
            flattened_data = np.log10(flattened_data)
    
    xout, yout, weights = loess_1d(np.array(flattened_domain), np.array(flattened_data), frac=frac)
    p = sorted(range(len(xout)),key=lambda i: xout[i])
    xout = xout[p]
    yout = yout[p]
    
    if logFilter:

        if logFilterAx == "both" or logFilterAx == "x":
            xout = [10.0**x for x in xout]

        if logFilterAx == "both" or logFilterAx == "y":
            yout = [10.0**y for y in yout]
 

    if ax is None:
        return xout, yout
    else:
        ax.plot(xout,yout,**kwargs)



def dual_faced_violin_plots(x_loc,data1,data2,ax,colors1,colors2,**kwargs):

    v1 = ax.violinplot(data1, points=100, positions=x_loc, showmeans=False, showextrema=False, 
                       showmedians=False,**kwargs)#,quantiles=[[.2,.8]]*len(data2))
    for (b,c) in zip(v1['bodies'],colors1):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_facecolor(c)
        b.set_edgecolor(c)
        #b.set_alpha(1)
        b.set_color(c)

    v2 = ax.violinplot(data2,positions = x_loc, points=100,#positions=np.arange(0, len(data2))
                showmeans=False, showextrema=False, showmedians=False,**kwargs)#,quantiles=[[.2,.8]]*len(data2))

    for (b,c) in zip(v2['bodies'],colors2):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_facecolor(c)
        b.set_edgecolor(c)
        #b.set_alpha(1)
        b.set_color(c)

def bold_outlined_text(ax,text_label,color,loc):
    text = ax.annotate(text_label,xy=loc, xycoords='axes fraction', c=color)
    text.set_path_effects([path_effects.Stroke(linewidth=.5, foreground='black'),
                       path_effects.Normal()])

#from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

#import numpy as np
#
# src: https://stackoverflow.com/questions/53849636/draw-a-double-box-plot-chart-2-axes-box-plot-box-plot-correlation-diagram-in

def boxplot_2d(x,y, ax,color, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = color,
        color=color,
        zorder=3,
        alpha=.5
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color=color,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color=color,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color=color, marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min([data for data in x if data > (xlimits[0]-whis*iqr)])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max([data for data in x if data < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    print(xlimits,ylimits)
    #print(ylimits[0],whis,iqr)
    #print(ylimits[0]-whis*iqr)

    bottom = [data for data in y if data  > ylimits[0]-whis*iqr]
    if len(bottom) == 0:
        pass
    else:
        bottom = np.min(bottom)
        print(bottom)
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
            color = color,
            zorder = 1
        )
        ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [bottom, bottom], 
            color = color,
            zorder = 1
        )
        ax.add_line(whisker_bar)
   

    ##top
    top = np.max([data for data in y if data < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    #mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    #ax.scatter(
    #    x[mask],y[mask],
    #    facecolors='none', edgecolors=color
    #)


#
#  Annotation Functions
#

def underlined_annotation(f,ax,xy,text,linestyle=None,**annotate_kwargs):

    #f = plt.gcf()
    text = ax.annotate(text,xy=xy,**annotate_kwargs)

    tb = text.get_tightbbox(f.canvas.get_renderer()).transformed(f.transFigure.inverted())
    ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),xycoords="figure fraction",
                arrowprops=dict(arrowstyle="-", color='k',linestyle=linestyle))
