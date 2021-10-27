import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_subplot_grid(iobjs,figsize=(18,6),dpi=80,
                 grid_shape=(2,3),
                     subplots_margins={"left":0.125,"right":0.9,
        "bottom":0.1,"top":0.9,"wspace":0.2 ,"hspace":0.7 }):
    fig = plt.figure(figsize=figsize,dpi=dpi)
    n = len(iobjs)
    axes = []
    for i in range(n):
        obj = iobjs[i]
        grid = obj['grid']
        rowspan,colspan = grid.get('rowspan',1),grid.get('colspan',1)
        axx = plt.subplot2grid(grid_shape,grid['position'],rowspan=rowspan,colspan=colspan)
        axes.append(axx)
    #
    for i in range(n):
        obj = iobjs[i]
        ax = axes[i]
        tdf = obj['df'].copy()
        cmap = obj.get("cmap","Blues")
        cbar_kws = obj.get("cbar_kws",{})
        title = obj.get("title","")
        xlabel = obj.get("xlabel","")
        ylabel = obj.get("ylabel","")
        xticks_size = obj.get("xticks_size",12)
        yticks_size = obj.get("yticks_size",12)
        xlabel_size = obj.get("xlabel_size",12)
        ylabel_size = obj.get("ylabel_size",12)
        plot_type = obj.get('plot_type','heatmap')
        if plot_type=="table":
            annot = tdf.values
            fmt = "d"
            if "." in f"{annot.flatten()[0]}":
                fmt = ".2f"
            tdf = pd.DataFrame(data=np.zeros_like(annot),columns=tdf.columns,index=tdf.index)
            sns.heatmap(tdf,cbar=False,cmap="Blues",annot=annot,fmt=fmt,
                       linewidths=1, linecolor='black',
                       ax=ax)
            ax.xaxis.tick_top() # x axis on top
            ax.xaxis.set_label_position('top')
        elif plot_type=="heatmap":
            sns.heatmap(tdf,cmap=cmap,ax=ax,cbar_kws=cbar_kws)
        elif plot_type=="line":
            index = tdf.index.to_list()
            columns = tdf.columns.to_list()
            xs = [ _ for _ in range(len(columns))]
            plt.plot(xs,tdf.values.transpose())
            plt.legend(index)
            ax.set_xticks(xs)
            ax.set_xticklabels(columns,rotation=90)
        plt.subplots_adjust(**subplots_margins)
        ax.set_title(title,fontweight='bold')
        ax.set_xlabel(xlabel,fontsize=xlabel_size)
        ax.set_ylabel(ylabel,fontsize=ylabel_size)
        ax.tick_params(axis='x', labelsize=xticks_size)
        ax.tick_params(axis='y', labelsize=yticks_size)
