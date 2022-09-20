import matplotlib.pyplot as plt
import random

from AStar import AStar
from City import City
from Plot import Plot

def read_coordiantes(filename: str) -> dict[str, City]:
    """
    Read the coordinates and city names from a file.

    line attributes: city_name latitude longitude
    example line: Attica 37.2421271 -98.2351967
    """

    with open(filename, "r") as f:
        lines = f.readlines()
        
    cities: dict[str, City] = {}
    for line in lines:
        name, latitude, longitude = line.split()
        cities[name] = City(name, float(latitude), float(longitude))

    return cities

def read_adjacencies(filename: str, cities: dict[str, City]) -> dict[str, City]:
    """
    Read the adjancencies from a file.
    Each line contains a city in the first position and all the cities adjacent 
    to it.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.split()

        root: City = cities[line[0]]
        roots_adjacencies: list[str] = [
            cities[name]
            for name 
            in line[1:]
            if name in cities  
        ]
        # re: if name in cities
        # doesn't make sense that a city can be adjancent but coords are not known

        # add the adjacencies to the root city and add the root to the adjancencies' adjacencies for redundancy
        # if the adjacency list was perfect, we wouldn't have to add the root to the adjacency's adjacencies dict
        for adjacency in roots_adjacencies:
            root.adjacencies[adjacency.name] = adjacency
            adjacency.adjacencies[root.name] = root
            
    return cities

def main() -> None:

    cities = read_coordiantes("coordinates.txt")
    cities = read_adjacencies("Adjacencies.txt", cities)

    seed: float = random.randint(0, 99999)
    # seed = 65204
    random.seed(seed)

    # randomly pick start and goal
    start: City = random.choice(list(cities.values()))
    while True:
        goal: City = random.choice(list(cities.values()))
        if start != goal:
            break

    aStar: AStar = AStar(start, goal)
    aStar.find_path()

    fig, (full_view_plot, close_up_plot) = plt.subplots(1, 2)
    fig.suptitle(f"Guided search using A* (seed: {seed})")

    # full_view_plot
    full_view_plot.set_title("Overview")
    Plot.setup_plot(full_view_plot, cities.values())
    Plot.set_grid_bounds(full_view_plot, cities.values())
    Plot.plot_cities(
        full_view_plot, 
        [city for city in cities.values() if city not in aStar.path], 
        color="blue"
    )
    Plot.plot_cities(
        full_view_plot, 
        [city for city in aStar.closed_set if city not in aStar.path], 
        color="black"
    )
    Plot.plot_path(full_view_plot, aStar.path)

    # close_up_plot
    close_up_plot.set_title("Close-up of Path")
    Plot.setup_plot(close_up_plot, cities.values())
    Plot.set_grid_bounds(close_up_plot, aStar.closed_set)
    Plot.plot_cities(
        close_up_plot, 
        [city for city in aStar.closed_set if city not in aStar.path], 
        color="black"
    )
    Plot.plot_path(close_up_plot, aStar.path)
    Plot.legend()

    plt.show()
    fig.savefig("AStar.png")

if __name__ == "__main__":
    main()