import math
import operator
import collections

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()


def euclideanDistance(instance1, instance2, length):
	distance = 0

	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)

	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1

	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))

	distances.sort(key=operator.itemgetter(1))
	neighbors = []

	for x in range(k):
		neighbors.append(distances[x][0])

	return neighbors

def mean_shift_distance(data, cx, cy, i_centroid, cluster_labels):
        distances = []
        features = []
        for (x,y) in data[cluster_labels == i_centroid]:
            distance = np.sqrt((x-cx)**2+(y-cy)**2)
            feature = cy
            distances.append(distance)
            features.append(feature)
        
        #distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
        return distances, features

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten1(data, group_names):
    try:
        group, group_names = group_names[0], group_names[1:]
    except IndexError:
        # No more key to extract, we just reached the most nested dictionnary
        yield data.copy()  # Build a new dict so we don't modify data in place
        return  # Nothing more to do, it is already considered flattened

    try:
        for key, value in data.iteritems():
            # value can contain nested dictionaries
            # so flatten it and iterate over the result
            for flattened in flatten(value, group_names):
                flattened.update({group: key})
                yield flattened
    except AttributeError:
        yield {group: data}

if __name__ == '__main__':
    '''
    g = Graph()
    g.add_vertex('a')
    g.add_vertex('b')
    g.add_edge('a', 'b', 7)
    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print '( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w))

    for v in g:
        print 'g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()])
    '''