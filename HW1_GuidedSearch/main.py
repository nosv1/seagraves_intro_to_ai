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

def get_city_from_user(cities: dict[str, City], start_goal: str) -> City:
    """
    Get the city from the user, if the city is not in the list, ask again
    """

    city: City = None
    while not city:
        try:
            city = input(f"Enter the number for the {start_goal} city: ")
            return cities[list(cities.keys())[int(city)-1]]
        except (ValueError, IndexError):
            print("Invalid input.")
            city = None

def main() -> None:
    ############################################################################
    # Read input files
    
    cities: dict[str, City] = read_coordiantes("coordinates.txt")
    cities = read_adjacencies("Adjacencies.txt", cities)

    ############################################################################
    # Get the start and end cities from the user

    print("#" * 60)
    print("Available cities:")
    for i, city in enumerate(cities.values()):
        print(str(f"{i+1}. {city.name}").ljust(20), end="")
        if (i+1) % 3 == 0:
            print()
    print("#" * 60)

    start_city: City = get_city_from_user(cities, "start")
    goal_city: City = None
    while not goal_city:
        goal_city: City = get_city_from_user(cities, "goal")
        if goal_city == start_city:
            print("Start and goal cities cannot be the same.")
            goal_city = None

    print(f"Distance between start and goal cities: {start_city.distance(goal_city):.2f} miles")

    ############################################################################
    # Random start and goal cities

    # seed: float = random.randint(0, 99999)
    # # seed = 65204
    # random.seed(seed)

    # # randomly pick start and goal
    # start_city: City = random.choice(list(cities.values()))
    # while True:
    #     goal_city: City = random.choice(list(cities.values()))
    #     if start_city != goal_city:
    #         break

    ############################################################################
    # Run the A* algorithm

    aStar: AStar = AStar(start_city, goal_city)
    aStar.find_path()

    print(f"Path distance: {aStar.path[0].total_cost:.2f} miles")

    ############################################################################
    # Plot the path

    fig, (full_view_plot, close_up_plot) = plt.subplots(1, 2)
    fig.suptitle(
        f"Guided search using A*\n"
        f"Path Distance: {aStar.path[0].total_cost:.2f} miles"
    )

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

    # add image to full view plot
    full_view_plot.imshow(plt.imread("Kansas Satellite.png"), origin="upper", extent=[
        fig.get_axes()[0].get_xlim()[0],
        fig.get_axes()[0].get_xlim()[1],
        fig.get_axes()[0].get_ylim()[0],
        fig.get_axes()[0].get_ylim()[1]
    ])

    plt.show()
    fig.savefig("AStar.png")

if __name__ == "__main__":
    main()