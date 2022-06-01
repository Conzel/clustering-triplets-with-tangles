from pathlib import Path
from typing import Optional

import altair as alt
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

CB_COLOR_CYCLE = ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF"]
LINE_STYLES = ["-o", "-^", "-s", 
               "-*", "-D", 
               "--o", "--^", "--s", "--*"]
SYMBOLS = ["o", "^", "s", "*", "D"]
MARKER_SIZES = [7,7,7, 
                9,7,7,
                7,7,9]

class ThesisStyleCycler:
    """
    Cycles through a list of plotting styles suitable for a thesis 
    (as we have a colorblind color scheme and line-styles that
    are suitable for a black-grey printer).

    It is already initialized with a valid style and color.
    Access is through the respective properties.
    """
    def __init__(self) -> None:
        self._state = 0
        self._color = CB_COLOR_CYCLE[0]
        self._line_style = LINE_STYLES[0]
        self._marker_size = MARKER_SIZES[0]

    def next(self):
        self._state += 1
        self._color = CB_COLOR_CYCLE[self._state % len(CB_COLOR_CYCLE)]
        self._line_style = LINE_STYLES[self._state % len(LINE_STYLES)]
        self._marker_size = MARKER_SIZES[self._state % len(MARKER_SIZES)]
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def line_style(self) -> str:
        return self._line_style

    @property
    def marker_size(self) -> int:
        return self._marker_size

class Plotter():
    def __init__(self, results_folder=None):
        if results_folder is None:
            self.results_folder = None
        else:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(parents=True, exist_ok=True)

    def scatter(self, data):
        """
        Plots the data points. Data is a numpy array with columns x, y.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def assignments(self, data, labels):
        """
        Plots the assignments of the data points to the clusters.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def comparison(self, data, true, predicted):
        """
        Compares predicted to true label.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def show(self):
        """
        Shows the plot.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def save(self):
        """
        Saves the plot.
        """
        raise NotImplementedError("Not implemented for this plotter.")


class AltairPlotter(Plotter):
    def __init__(self, results_folder=None):
        super().__init__(results_folder=results_folder)
        self.last_chart = None

    # Decorator for cleanup
    def _altair_plotting_function(func):
        def wrapper(self, *args, **kwargs):
            chart = func(self, *args, **kwargs)
            self.last_chart = chart
            return chart
        return wrapper

    @_altair_plotting_function
    def parameter_variation(self, df, attribute_name):
        """
        Plots the values of ars and nmi over a range of the given attribute.
        Also plots baseline if applicable
        """
        df = df.sort_values(by=[attribute_name]).groupby(
            [attribute_name, "kind"]).mean().reset_index()
        nmi = alt.Chart(df).mark_line(
            point={
                "filled": False,
                "fill": "white"
            }).encode(
            x=attribute_name,
            y="nmi",
            color="kind:N"
        ).interactive()
        ars = alt.Chart(df).mark_line(
            point={
                "filled": False,
                "fill": "white"
            }).encode(
                x=attribute_name,
                y="ars",
                color="kind:N"
        ).interactive()
        return nmi | ars

    @_altair_plotting_function
    def scatter(self, data):
        """
        Plots the data points.
        """
        data = pd.DataFrame(data, columns=["x", "y"])

        chart = alt.Chart(data).mark_point().encode(
            x="x",
            y="y",
        )
        return chart

    @_altair_plotting_function
    def assignments(self, data: np.ndarray, labels: np.ndarray) -> alt.Chart:
        """
        Plots the assignments of the data points to the clusters.
        """
        data_df = pd.DataFrame(data, columns=["x", "y"])
        data_df["predicted"] = labels

        chart = alt.Chart(data_df).mark_point().encode(
            x="x",
            y="y",
            color=alt.Color("predicted:N", scale=alt.Scale(scheme="dark2"))
        )
        return chart

    @_altair_plotting_function
    def comparison(self, data, true, predicted):
        """
        Compares predicted to true label.
        """
        chart1 = self.assignments(data, true).properties(title="True labels")
        chart2 = self.assignments(data, predicted).properties(
            title="Predicted labels")
        combined = chart1 | chart2
        combined = combined.properties(
            title=f"Clustering comparison, NMI = {normalized_mutual_info_score(true, predicted):.2f}")
        return combined

    def show(self):
        """
        Shows the plot.
        """
        if self.last_chart is not None:
            self.last_chart.show()
        else:
            raise ValueError("No chart to show.")

    def save(self, name):
        if self.last_chart is None:
            raise ValueError("No chart to show.")
        if self.results_folder is None:
            raise ValueError("No results folder to save to.")
        self.last_chart.save(str(self.results_folder / name))

class ThesisPlotter:
    """
    Contains all the functions for plotting the figures in the thesis.
    """
    def __init__(self, results_folder: Optional[Path] = None, methods_plotted: Optional[set[str]] = None, style_dict: Optional[dict[str, tuple[str, str, int]]] = None) -> None:
        self.results_folder = results_folder
        self.methods_plotted = methods_plotted
        self.style_dict = style_dict

    def labels_to_colors(self, xs: np.ndarray) -> list[tuple]:
        cmap = get_cmap("tab10")
        return [cmap(x) for x in xs]
    
    
    def assignments(self, xs: np.ndarray, ys: np.ndarray, alternative_symbols: bool = False):
        """
        Assumes contiguous labels y.
        """
        plt.figure()
        for i in range(np.max(ys.astype(int)) + 1):
            mask = (ys == i)
            if alternative_symbols:
                symbol = SYMBOLS[i % 5]
            else:
                symbol = "."
            plt.plot(xs[:, 0][mask], xs[:, 1][mask], symbol, label="Cluster " + str(i + 1))
        plt.xlabel("x")
        plt.ylabel("y")
    
    def assignments_different_symbols(self, xs: np.ndarray, ys: np.ndarray):
        plt.figure()

        for i in range(np.max(ys.astype(int)) + 1):
            mask = (ys == i)
            plt.plot(xs[:, 0][mask], xs[:, 1][mask], SYMBOLS[i % 5], c=CB_COLOR_CYCLE[i] label="Cluster " + str(i + 1))
        plt.xlabel("x")
        plt.ylabel("y")
    
    
    def line(self, df, x: str, y: str, methods_to_use: Optional[set[str]] = None):
        plt.figure()
        df = df.groupby(["method", x]).mean().reset_index()
        methods = set(df.method.unique())
        if methods_to_use is None:
            methods_to_use = self.methods_plotted
        if methods_to_use is not None:
            methods = methods & methods_to_use
        
        style_cycler = ThesisStyleCycler()
        for method in methods:
            if self.style_dict is not None:
                line_style, color, marker_size = self.style_dict[method]
            else:
                line_style, color, marker_size = style_cycler.line_style, style_cycler.color, style_cycler.marker_size
                style_cycler.next()
            x_arr = df[df.method == method][x]
            y_arr = df[df.method == method][y]
            if x_arr.size != 0 and y_arr.size != 0:
                plt.plot(x_arr, y_arr, line_style, c=color, markersize=marker_size, label=f"{method}")
        plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)
    
    
    def heatmap(self, df, x1: str, x2: str, y: str, method: str):
        plt.figure()
        df = df[df.method == method].groupby(
            [x1, x2]).mean().reset_index().sort_values([x1, x2])
        x1v, x2v = np.meshgrid(df[x1].unique(), df[x2].unique(), indexing="ij")
        yv = np.zeros_like(x1v)
        for i in range(x1v.shape[0]):
            for j in range(x1v.shape[1]):
                yv[i, j] = df[(df[x1] == x1v[i, j]) & (df[x2] == x2v[i, j])][y]
        plt.pcolormesh(x1v, x2v, yv, cmap="Blues")
        plt.colorbar()
        plt.clim(0.0, 1.0)
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.xlim([x1v.min(), x1v.max()])
        plt.ylim([x2v.min(), x2v.max()])
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xscale("log")
    
    
    def save(self,name: str):
        """
        Pass as name just the stem, no extension, no results folder appended.
        Saves the plot on the current graphical axis under the given name.
        """
        if self.results_folder is not None:
            plt.savefig(self.results_folder / f"{name}.pgf")
            plt.savefig(self.results_folder / f"{name}.png")
            plt.savefig(self.results_folder / f"{name}.pdf")
        else:
            plt.savefig(f"{name}.pgf")
            plt.savefig(f"{name}.png")
            plt.savefig(f"{name}.pdf")
    
def save_all(folder_name, figure_name):
    plt.savefig(Path(folder_name) / f"{figure_name}.pgf")
    plt.savefig(Path(folder_name) / f"{figure_name}.pdf")
    plt.savefig(Path(folder_name) / f"{figure_name}.png")
    
    