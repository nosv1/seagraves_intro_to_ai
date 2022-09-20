import matplotlib.pyplot as plt

from City import City

class Plot:
    def setup_plot(ax: plt.Axes, cities: list[City]) -> None:
        ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)')

    def set_grid_bounds(
        ax: plt.Axes, cities: list[City]
    ) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        for city in cities:
            min_x = min(min_x, city.longitude)
            max_x = max(max_x, city.longitude)
            min_y = min(min_y, city.latitude)
            max_y = max(max_y, city.latitude)

        ax.set_xlim(min_x - .25, max_x + .25)
        ax.set_ylim(min_y - .25, max_y + .25)

    def plot_cities(ax: plt.Axes, cities: list[City], color: str) -> None:
        for city in cities:
            ax.text(
                city.longitude, city.latitude, city.name, 
                fontsize=6, ha="center", va="center", color="white",
                bbox=dict(boxstyle="round", fc=color)
            )

    def plot_path(ax: plt.Axes, path: list[City]) -> None:
        latitudes: list[float] = [city.latitude for city in path]
        longitudes: list[float] = [city.longitude for city in path]
        ax.plot(longitudes, latitudes, color='blue', label='path')
        for i, city in enumerate(path):
            ax.text(
                city.longitude, 
                city.latitude, 
                city.name, 
                fontsize=8, 
                ha="center", 
                va="center", 
                bbox=dict(boxstyle="round", fc="w" if not i else "grey")
            )

    def legend():
        plt.legend(
            ncol=1,
            fancybox=True,
            shadow=True,
            handles=[
                plt.Line2D([0], [0], marker='o', color='black', label='start city', markerfacecolor='white', markersize=10),
                plt.Line2D([0], [0], marker='o', color='black', label='cities on path', markerfacecolor='grey', markersize=10),
                plt.Line2D([0], [0], marker='o', color='white', label='adjancent cities to path', markerfacecolor='black', markersize=10),
                plt.Line2D([0], [0], marker='o', color='white', label='all cities', markerfacecolor='blue', markersize=10),
                plt.Line2D([0], [0], color='blue', lw=2, label='path'),
            ]
        )