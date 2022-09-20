from City import City

class AStar:
    def __init__(self, start: City, goal: City) -> None:
        self.start = start
        self.goal = goal

        self._current_city: City = start
        self._open_set: dict[str, City] = {
            self._current_city.name: self._current_city
        }
        self._closed_set: dict[str, City] = {}
        self._path: list[City] = []

    @property
    def closed_set(self) -> list[City]:
        return self._closed_set.values()

    @property
    def path(self) -> list[City]:
        return self._path

    def add_adjacencies_to_open_set(self) -> None:
        for adjacency in self._current_city.adjacencies.values():
            temp_adjacency: City = adjacency.copy()
            temp_adjacency.start_to_city_cost = (
                self._current_city.start_to_city_cost +
                self._current_city.distance(adjacency)
            ) 
            temp_adjacency.city_to_goal_cost = adjacency.distance(self.goal)
            temp_adjacency.parent = self._current_city

            # adjacency already in open set, update cost and parent as needed
            if adjacency.name in self._open_set:
                if temp_adjacency.total_cost < adjacency.total_cost:
                    self._open_set[adjacency.name] = temp_adjacency

            # adjacency not in open set and not in closed set, add it to open set
            elif adjacency.name not in self._closed_set:
                self._open_set[adjacency.name] = temp_adjacency

    def find_path(self) -> list[City]:

        # while current city is not goal
        while self._current_city != self.goal:

            # get new city with lowest total cost
            self._current_city = min(
                self._open_set.values(),
                key=lambda city: city.total_cost
            )

            del self._open_set[self._current_city.name]

            self.add_adjacencies_to_open_set()

            self._closed_set[self._current_city.name] = self._current_city

        # we've found goal

        self._closed_set[self._current_city.name] = self._current_city

        # build the path, goal to start
        self._path = [self._closed_set[self.goal.name]]
        while self._path[-1].parent:
            self._path.append(self._closed_set[self._path[-1].parent.name])