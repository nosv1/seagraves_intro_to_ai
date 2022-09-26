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
        """
        - for each adjacency
            - copy adjacency to temp city
            - get cost from current city to temp city
            - get cost from temp city to goal
            - set parent of temp city to current city
            - if temp city is in open set
                - if temp city has lower total cost
                    - update temp city in open set
            - if temp city is not in closed set
                - add temp city to open set
        """

        for adjacency in self._current_city.adjacencies.values():
            # copy adjacency to temp city
            temp_adjacency: City = adjacency.copy()

            # set the costs of temp city
            temp_adjacency.start_to_city_cost = (
                self._current_city.start_to_city_cost +
                self._current_city.distance(adjacency)
            ) 
            temp_adjacency.city_to_goal_cost = adjacency.distance(self.goal)

            # set the parent
            temp_adjacency.parent = self._current_city

            # adjacency already in open set, update cost and parent as needed
            if adjacency.name in self._open_set:
                if temp_adjacency.total_cost < adjacency.total_cost:
                    self._open_set[adjacency.name] = temp_adjacency

            # adjacency not in open set and not in closed set, add it to open set
            elif adjacency.name not in self._closed_set:
                self._open_set[adjacency.name] = temp_adjacency

    def find_path(self) -> None:
        """
        - while current city is not goal
            - get new city with lowest total cost from open set (F(n) = G(n) + H(n))
            - remove current city from open set
            - add current city to closed set
            - add adjacencies to open set
        - build path from goal to start
        """

        # while current city is not goal
        while self._current_city != self.goal:

            # get new city with lowest total cost
            self._current_city = min(
                self._open_set.values(),
                key=lambda city: city.total_cost
            )

            # remove current city from open set
            del self._open_set[self._current_city.name]

            # add current city to closed set
            self._closed_set[self._current_city.name] = self._current_city

            # add adjacencies to open set
            self.add_adjacencies_to_open_set()

        # goal has been found

        # add current city to closed set for the sake of bookkeeping
        self._closed_set[self._current_city.name] = self._current_city

        # build the path, goal to start
        self._path = [self._closed_set[self.goal.name]]
        while self._path[-1].parent:
            self._path.append(self._closed_set[self._path[-1].parent.name])