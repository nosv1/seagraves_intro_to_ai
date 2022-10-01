from GeneticAlgorithm.Gene import Gene

class Room(Gene):
    def __init__(self, building: str, room: str, capacity: int) -> None:
        super().__init__()
        self.building = building
        self.room = room
        self.capacity = capacity

    @property
    def key(self) -> str:
        return f"{self.building}-{self.room}"

    def parse_key(key: str) -> tuple[str, int]:
        building, room = key.split("-")
        return building, int(room)