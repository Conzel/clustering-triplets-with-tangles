from utils import flatten, hierarchy_list_flatmap, hierarchy_list_map, index_cluster_list


def test_flatten():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten([1, 2, 3]) == [1, 2, 3]


def test_cluster_list_index_transform():
    assert index_cluster_list([[0, 2], [1, 3], [4]]) == [0, 1, 0, 1, 2]


def test_hierarchy_list_map():
    assert hierarchy_list_map(
        [[[0, 1], [2, 3]], [3, 4]], lambda x: x + 1) == [[[1, 2], [3, 4]], [4, 5]]


def test_hierarchy_list_flatmap():
    assert hierarchy_list_flatmap([[0, 1], 3], lambda x: [
                                  x, x + 1]) == [[0, 1, 1, 2], 3, 4]
