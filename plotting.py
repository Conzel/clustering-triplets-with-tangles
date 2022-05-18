from pathlib import Path
from typing import Optional

import altair as alt
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


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
    def __init__(self, results_folder: Optional[Path] = None) -> None:
        self.results_folder = results_folder

    def labels_to_colors(self, xs: np.ndarray) -> list[tuple]:
        cmap = get_cmap("tab10")
        return [cmap(x) for x in xs]
    
    
    def assignments(self, xs: np.ndarray, ys: np.ndarray):
        """
        Assumes contiguous labels y.
        """
        plt.figure()
        for i in range(np.max(ys.astype(int)) + 1):
            mask = (ys == i)
            plt.plot(xs[:, 0][mask], xs[:, 1][mask], ".", label="Cluster " + str(i))
        plt.xlabel("x")
        plt.ylabel("y")
    
    
    def line(self, df, x: str, y: str, methods_to_use: Optional[set[str]] = None):
        plt.figure()
        df = df.groupby(["method", x]).mean().reset_index()
        methods = set(df.method.unique())
        if methods_to_use is not None:
            methods = methods ^ methods_to_use
        for method in methods:
            x_arr = df[df.method == method][x]
            y_arr = df[df.method == method][y]
            plt.plot(x_arr, y_arr, "--o", label=f"{method}")
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
    
    