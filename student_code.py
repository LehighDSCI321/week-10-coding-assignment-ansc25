"Week 10 coding assignment TraversableDigraph and DAG - Submitted by Anurag Subedi"
from collections import deque
from graphviz import Digraph
from bokeh.plotting import figure, show, output_file, save

class VersatileDigraph():
    """ A versatile directed graph implementation.
    Nodes are identified by string IDs and may carry a numeric value.
    Edges are from a start node to an end node, have a numeric weight,
    and have a string name that is unique among the edges leaving a start node.
    """
    def __init__(self):
        """Initialize empty graph structures."""
        self._nodes = {}
        self._out_edges = {}
        self._out_names = {}

    def _ensure_node_exists(self, node_id):
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")

    def predecessors(self, node_id):
        """Returns the predecessors for a given node_id"""
        self._ensure_node_exists(node_id)
        return [k for k, v in self._out_edges.items() if v.get(node_id) is not None]

    def successors(self, node_id):
        """Returns the predecessors for a given node_id"""
        self._ensure_node_exists(node_id)
        return [k for k in self._out_edges.get(node_id).keys() if k is not None]

    def successor_on_edge(self, node_id, edge_name):
        """Return the successsor based on edge_name and node_id"""
        self._ensure_node_exists(node_id)

        if edge_name not in self._out_names.get(node_id, {}):
            raise KeyError(f"Edge named '{edge_name}' does not exist from start node '{node_id}'.")
        return self._out_names[node_id][edge_name]

    def in_degree(self, node_id):
        """Return the indegree of a given node"""
        self._ensure_node_exists(node_id)
        return len([k for k, v in self._out_edges.items() if v.get(node_id) is not None])

    def out_degree(self, node_id):
        """Return the outdegree of a given node"""
        self._ensure_node_exists(node_id)
        return len([k for k in self._out_edges.get(node_id).keys() if k is not None])

    def add_node(self, node_id, node_value=0):
        """Add a node with optional numeric value. If already present, update value.
        Raises TypeError if node_value is not numeric
        """
        if not isinstance(node_value, (int, float)):
            raise TypeError("Node value must be a numeric type (int or float).")

        if node_id in self._nodes:
            self._nodes[node_id] = node_value
            return
        self._nodes[node_id] = node_value
        self._out_edges.setdefault(node_id, {})
        self._out_names.setdefault(node_id, {})

    def add_edge(self, start_node_id, end_node_id,
                 start_node_value=None, end_node_value=None,
                 edge_name=None, edge_weight=0):
        """Add an edge from start_node_id to end_node_id."""
        if start_node_value is not None and not isinstance(start_node_value, (int, float)):
            raise TypeError("start_node_value must be numeric (int or float) if provided.")
        if end_node_value is not None and not isinstance(end_node_value, (int, float)):
            raise TypeError("end_node_value must be numeric (int or float) if provided.")

        if not isinstance(edge_weight, (int, float)):
            raise TypeError("edge_weight must be numeric (int or float).")
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")

        if start_node_id not in self._nodes:
            self.add_node(start_node_id, start_node_value or 0)
        elif start_node_value is not None:
            self._nodes[start_node_id] = start_node_value

        if end_node_id not in self._nodes:
            self.add_node(end_node_id, end_node_value or 0)
        elif end_node_value is not None:
            self._nodes[end_node_id] = end_node_value

        if edge_name is None:
            edge_name = f"{start_node_id}_to_{end_node_id}"

        self._out_edges[start_node_id][end_node_id] = (edge_weight, edge_name)
        self._out_names[start_node_id][edge_name] = end_node_id

    def get_nodes(self):
        """Return a list of node ids in the graph."""
        return list(self._nodes.keys())

    def get_edge_weight(self, start_node_id, end_node_id):
        """Return the weight of the edge from start_node_id to end_node_id."""
        if start_node_id not in self._out_edges:
            raise KeyError(f"Start node '{start_node_id}' does not exist.")
        entry = self._out_edges[start_node_id].get(end_node_id)
        if entry is None:
            raise KeyError(f"Edge from '{start_node_id}' to '{end_node_id}' does not exist.")
        weight, _ = entry
        return weight

    def get_node_value(self, node_id):
        """Return the numeric value of the node. Raises KeyError if missing."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return self._nodes[node_id]

    def print_graph(self):
        """Print readable descriptions of nodes and edges."""
        for node_id, value in self._nodes.items():
            print(f"Node {node_id} with value {value}")
        for start, ends in self._out_edges.items():
            for end, (weight, name) in ends.items():
                print(f"Edge from {start} to {end} with weight {weight} and name {name}")

    def plot_graph(self, filename=None, view=False):
        """Create a Graphviz Digraph of the current graph."""
        dot = Digraph(format="png")
        for node_id, value in self._nodes.items():
            dot.node(node_id, label=f"{node_id}: {value}")

        for start, ends in self._out_edges.items():
            for end, (weight, name) in ends.items():
                dot.edge(start, end, label=f"{name}: {weight}")

        if filename:
            dot.render(filename, view=view)
        return dot

    def plot_edge_weights(self, show_plot=False, output_html=None):
        """Create a bar plot (Bokeh) of edge weights."""

        labels = []
        weights = []
        edge_names = []
        for start, ends in self._out_edges.items():
            for end, (weight, name) in ends.items():
                labels.append(f"{start}→{end}")
                weights.append(weight)
                edge_names.append(name)

        if not labels:
            raise ValueError("Graph contains no edges to plot.")

        p = figure(x_range=labels, height=350, title="Distance in miles between city pairs",
                   toolbar_location=None, tools="")
        p.vbar(x=labels, top=weights, width=0.8)
        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.xaxis.major_label_orientation = 1.2

        if output_html:
            output_file(output_html)
            save(p)

        if show_plot:
            show(p)

        return p

class SortableDigraph(VersatileDigraph):
    """A directed graph with topological sorting capability.
    Extends VersatileDigraph with a top_sort method that returns
    a topologically sorted list of nodes using the counting-based algorithm.
    """

    def top_sort(self):
        """Return a topologically sorted list of nodes in the graph.
        
        Uses a counting-based iterative algorithm that:
        1. Counts in-degrees for all nodes
        2. Starts with nodes that have zero in-degree
        3. Iteratively removes nodes and updates counts
        
        Returns:
            List of node IDs in topologically sorted order
            
        Raises:
            ValueError: If the graph contains a cycle (not a DAG)
        """
        # Get all nodes in the graph
        nodes = self.get_nodes()

        # Initialize in-degree count for each node to 0
        count = {u: 0 for u in nodes}

        # Count in-degrees: for each node, count incoming edges
        for u in nodes:
            for v in self.successors(u):
                count[v] += 1

        # Q contains nodes with zero in-degree (valid starting nodes)
        q = [u for u in nodes if count[u] == 0]

        # S will contain the topologically sorted result
        s = []

        # Process nodes while we have valid start nodes
        while q:
            # Pick a node with zero in-degree
            u = q.pop()
            # Add it to the sorted result
            s.append(u)

            # For each successor of u, decrease its in-degree count
            for v in self.successors(u):
                count[v] -= 1
                # If v now has zero in-degree, it's a valid start node
                if count[v] == 0:
                    q.append(v)

        # If we didn't process all nodes, the graph has a cycle
        if len(s) != len(nodes):
            raise ValueError("Graph contains a cycle and cannot be topologically sorted.")

        return s

class TraversableDigraph(SortableDigraph):
    """A sortable directed graph with traversal capabilities.
    Extends SortableDigraph with DFS and BFS traversal methods.
    """

    def dfs(self, start_node):
        """Perform iterative depth-first search traversal from start_node.        
        Uses a stack (LIFO) to traverse the graph in depth-first order.  
        Args: start_node: The node ID to start traversal from  
        Yields: Node IDs in DFS order
        Raises: KeyError: If start_node does not exist in the graph
        """
        # Ensure the start node exists
        self._ensure_node_exists(start_node)

        s, q = set(), []  # Visited-set and stack (LIFO queue)
        q.append(start_node)  # We plan on visiting start_node

        while q:  # Planned nodes left?
            u = q.pop()  # Get one (LIFO - last in, first out)
            if u in s:  # Already visited? Skip it
                continue
            s.add(u)  # We've visited it now
            q.extend(self.successors(u))  # Schedule all neighbors
            # Only yield if it's not the start node
            if u != start_node:
                yield u  # Report u as visited

    def bfs(self, start_node):
        """Perform breadth-first search traversal from start_node.
        Uses a deque (FIFO) to traverse the graph in breadth-first order.        
        Args: start_node: The node ID to start traversal from
        Yields: Node IDs in BFS order
        Raises: : If start_node does not exist in the graph
        """
        # Ensure the start node exists
        self._ensure_node_exists(start_node)

        s = set()  # Visited-set
        q = deque()  # Queue (FIFO) for BFS
        q.append(start_node)  # We plan on visiting start_node

        while q:  # Planned nodes left?
            u = q.popleft()  # Get one (FIFO - first in, first out)
            if u in s:  # Already visited? Skip it
                continue
            s.add(u)  # We've visited it now
            for v in self.successors(u):  # Schedule all neighbors
                q.append(v)
            # Only yield if it's not the start node
            if u != start_node:
                yield u  # Report u as visited

class DAG(TraversableDigraph):
    """A Directed Acyclic Graph that prevents cycle creation.
    Extends TraversableDigraph and overrides add_edge to ensure
    that no edge addition will create a cycle in the graph.
    """

    def add_edge(self, start_node_id, end_node_id,
                 start_node_value=None, end_node_value=None,
                 edge_name=None, edge_weight=0):
        """Add an edge from start_node_id to end_node_id.
        
        Overrides the parent add_edge method to check for cycle creation.
        Before adding the edge, checks if there's already a path from
        end_node to start_node. If such a path exists, adding this edge
        would create a cycle, so an exception is raised.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            start_node_value: Optional value for start node
            end_node_value: Optional value for end node
            edge_name: Optional name for the edge
            edge_weight: Weight of the edge (default 0)
            
        """
        # First, ensure both nodes exist (add them if they don't)
        # We need to do this again before checking for cycles 
        if start_node_id not in self._nodes:
            self.add_node(start_node_id, start_node_value or 0)
        elif start_node_value is not None:
            self._nodes[start_node_id] = start_node_value

        if end_node_id not in self._nodes:
            self.add_node(end_node_id, end_node_value or 0)
        elif end_node_value is not None:
            self._nodes[end_node_id] = end_node_value

        # Check if adding this edge would create a cycle
        # A cycle would be created if there's already a path from end_node to start_node
        # We use DFS to check if we can reach start_node from end_node
        try:
            # Traverse from end_node and see if we can reach start_node
            for visited_node in self.dfs(end_node_id):
                if visited_node == start_node_id:
                    # Found a path from end_node to start_node!
                    # Adding this edge would create a cycle
                    raise ValueError(
                        f"Adding edge from '{start_node_id}' to '{end_node_id}' "
                        f"would create a cycle in the DAG."
                    )
        except KeyError:
            # end_node_id doesn't exist yet or has no successors
            # This is fine, no cycle possible
            pass

        # No cycle detected, safe to add the edge using parent's method
        super().add_edge(start_node_id, end_node_id,
                        start_node_value, end_node_value,
                        edge_name, edge_weight)

# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing TraversableDigraph - DFS and BFS")
    print("=" * 70)

    # Create a traversable graph
    tg = TraversableDigraph()

    # Build a sample graph
    #     A
    #    / \
    #   B   C
    #  / \   \
    # D   E   F

    tg.add_edge("A", "B")
    tg.add_edge("A", "C")
    tg.add_edge("B", "D")
    tg.add_edge("B", "E")
    tg.add_edge("C", "F")

    print("\nDFS from A:")
    dfs_result = list(tg.dfs("A"))
    print("  ", " -> ".join(dfs_result))

    print("\nBFS from A:")
    bfs_result = list(tg.bfs("A"))
    print("  ", " -> ".join(bfs_result))

    # test with the clothing example
    clothing_dag = DAG()

    # Add all dependencies
    print("\nBuilding clothing dependency DAG...")
    clothing_dag.add_edge("shirt", "pants")
    clothing_dag.add_edge("shirt", "vest")
    clothing_dag.add_edge("shirt", "jacket")
    clothing_dag.add_edge("pants", "shoes")
    clothing_dag.add_edge("pants", "belt")
    clothing_dag.add_edge("socks", "shoes")
    clothing_dag.add_edge("vest", "jacket")
    clothing_dag.add_edge("belt", "jacket")
    print("  All valid edges added")

    # Try to add an invalid edge
    print("\nadding jacket -> shirt (would create cycle):")
    try:
        clothing_dag.add_edge("jacket", "shirt")
        print("ERROR: Should have prevented cycle!")
    except ValueError as e:
        print("Correctly catched the cycle prevention exception")

    print("\nTopological sort (valid dressing order):")
    print("  ", " -> ".join(clothing_dag.top_sort()))

    print("\nDFS from 'shirt':")
    print("  ", " -> ".join(list(clothing_dag.dfs("shirt"))))

    print("\nBFS from 'shirt':")
    print("  ", " -> ".join(list(clothing_dag.bfs("shirt"))))
