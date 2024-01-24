class TrainingMarkerLabel:
    """
    We use these instead of just strings so that we can handle marker sets that use the same marker name across
    different skeletons in the training data to mean different things.

    This is only useful for handling the marker sets in the training data. For the marker sets at test time,
    we don't have this problem because we only have one skeleton, and one unified marker set.
    """
    name: str
    skeleton_index: int

    def __init__(self, name: str, skeleton_index: int):
        self.name = name
        self.skeleton_index = skeleton_index

    def __hash__(self):
        return hash((self.name, self.skeleton_index))

    def __eq__(self, other):
        return (self.name, self.skeleton_index) == (other.name, other.skeleton_index)

    def __repr__(self):
        return f'MarkerLabel(name={self.name}, skeleton_index={self.skeleton_index})'

    def __str__(self):
        return self.__repr__()
