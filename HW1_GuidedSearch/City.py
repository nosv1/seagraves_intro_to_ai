from __future__ import annotations

class City:
    def __init__(
        self, 
        name: str, 
        latitude: float, 
        longitude: float
    ) -> None:
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

        self.adjacencies: dict[str, City] = {}
        self.start_to_city_cost: float = 0
        self.city_to_goal_cost: float = 0
        self.parent: City = None

    @property
    def total_cost(self) -> float:
        return self.start_to_city_cost + self.city_to_goal_cost

    def __eq__(self, other: City) -> bool:
        return self.name == other.name

    def copy(self) -> City:
        """
        Returns a copy of the city
        """
        city: City = City(self.name, self.latitude, self.longitude)
        city.adjacencies = self.adjacencies
        city.start_to_city_cost = self.start_to_city_cost
        city.city_to_goal_cost = self.city_to_goal_cost
        city.parent = self.parent
        return city

    def distance(self, other: City) -> float:
        """
        Calculates the ecuclidean distance between two nodes
        """
        return (
            ((self.latitude - other.latitude) * 69) ** 2 + 
            ((self.longitude - other.longitude) * 54.6) ** 2
        ) ** 0.5