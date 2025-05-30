import pywavefront
import sys


def _load_vertices(path_obj_file):
    """
    Load the vertices of the obj file
    :param path_obj_file: path to the obj file
    :return: vertices
    """
    # Load the obj file
    obj = pywavefront.Wavefront(path_obj_file, strict=False)
    # Get the vertices of the obj file
    return obj.vertices


def _format_pin_ranges(pinned_indices):
    """
    Format the pin ranges as a list of lists

    :param pinned_indices: list of pinned indices
    :return: list of lists of pin ranges
    """
    pin_ranges = []
    start = pinned_indices[0]
    end = pinned_indices[0]
    for i in range(1, len(pinned_indices)):
        if pinned_indices[i] == end + 1:
            end = pinned_indices[i]
        else:
            pin_ranges.append([start, end])
            start = pinned_indices[i]
            end = pinned_indices[i]
    # Add the last range
    pin_ranges.append([start, end])
    return pin_ranges


def generate_pin_ranges_kite(path_obj_file):
    """
    Generate the pin ranges for the given obj file
    :param path_obj_file: path to the obj file
    :return: the pin ranges
    """
    vertices = _load_vertices(path_obj_file)
    biggest_x = -float('inf')
    for vertex in vertices:
        if abs(vertex[0]) > biggest_x:
            biggest_x = abs(vertex[0])

    pinned_indices = []
    for index, vertex in enumerate(vertices):
        if abs(vertex[0]) == biggest_x:
            pinned_indices.append(index)
        # Check if the vertex is on the y-axis (backbone)
        elif vertex[1] == 0:
            pinned_indices.append(index)

    if len(pinned_indices) == 0:
        raise ValueError("No pinned vertices found.")

    return _format_pin_ranges(pinned_indices)


def generate_pin_ranges_flying_squirrel(path_obj_file):
    """
    Generate the pin ranges for the given obj file
    :param path_obj_file: path to the obj file
    :return: the pin ranges
    """
    vertices = _load_vertices(path_obj_file)

    pinned_indices = []
    for index, vertex in enumerate(vertices):
        if vertex[1] == 0:
            pinned_indices.append(index)

    if len(pinned_indices) == 0:
        raise ValueError("No pinned vertices found.")

    return _format_pin_ranges(pinned_indices)


def generate_connects_flying_squirrel_in_range(path_obj_file, x_pos, min_y_pos, max_y_pos):
    """
        Generate the pin ranges for the given obj file
        :param path_obj_file: path to the obj file
        :param x_pos: x position
        :param min_y_pos: minimum y position
        :param max_y_pos: maximum y position
        :return: tuple of index and vertex
    """
    vertices = _load_vertices(path_obj_file)
    connects = []

    for index, vertex in enumerate(vertices):
        # round to 5 decimal to avoid floating point errors
        if round(vertex[0], 5) == round(x_pos, 5) and min_y_pos <= vertex[1] <= max_y_pos:
            connects.append((index, vertex))

    return connects


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage: python generate_pin_ranges_obj.py <obj>")
    obj_file = sys.argv[1]
    print(generate_pin_ranges_kite(obj_file))
