import math

from typing import Any

import torch

from typing_extensions import Self


class MCTSNode:
    """
    A node in the MCTS tree.

    Attributes:
        scene (torch.Tensor): shape (1, N, V). Unnormalized scene.
        inpainting_mask (torch.Tensor): shape (1, N, V). Boolean mask.
        parent (MCTSNode or None): Parent node.
        children (List[MCTSNode]): Child nodes.
        visits (int): Number of visits to this node.
        total_value (float): Sum of rewards from all simulations through this node.
        metadata (dict): Additional information about the node.
        node_id (str): Unique identifier for this node.
    """

    _next_id = 0

    def __init__(
        self,
        scene: torch.Tensor,
        inpainting_mask: torch.Tensor,
        parent: Self | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.scene = scene
        self.inpainting_mask = inpainting_mask
        self.parent = parent
        self.children = []

        self.visits = 0
        self.total_value = 0.0
        self.metadata = metadata or {}

        # Assign unique ID to this node
        self.node_id = str(MCTSNode._next_id)
        MCTSNode._next_id += 1

    @property
    def cost(self) -> float:
        """The cost of this node (lower is better)."""
        return self.inpainting_mask.sum().item()

    @property
    def value(self) -> float:
        """The average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def is_leaf(self) -> bool:
        """Whether this node is a leaf node (has no children)."""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Whether this node is a terminal node (perfect solution)."""
        return not self.inpainting_mask.any()

    def get_tree_data(self) -> dict[str, Any]:
        """
        Get data about this node and its descendants for visualization. It is
        recommended to call this function on the root node.

        Returns:
            dict[str, Any]: A dictionary containing two keys: 'nodes' and 'edges'.
                - 'nodes' is a list of dictionaries, each representing a node in the
                    MCTS tree with its unique identifier, cost, visit count, average
                    value, and terminal status.
                - 'edges' is a list of dictionaries representing the connections between
                    parent and child nodes, indicating the structure of the tree for
                    visualization.
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def traverse(node: Self) -> None:
            # Add this node.
            node_data = {
                "id": node.node_id,
                "cost": node.cost,
                "visits": node.visits,
                "value": node.value,
                "is_terminal": node.is_terminal(),
            }
            # Add metadata if available.
            for key, value in node.metadata.items():
                if isinstance(value, (int, float)):
                    node_data[key] = value

            nodes.append(node_data)

            # Add edges to children.
            for child in node.children:
                edges.append({"from": node.node_id, "to": child.node_id})
                traverse(child)

        traverse(self)
        return {"nodes": nodes, "edges": edges}


def select_node_with_uct(node: MCTSNode, exploration_weight: float) -> MCTSNode:
    """
    Select the child node with the highest UCB score.

    The UCT (Upper Confidence Bound for Trees) formula used is:
        𝑈𝐶𝑇(𝑗) = 𝑋̄ⱼ + 𝑐 ⋅ √(2 ⋅ ln(𝑁ᵢ) / 𝑛ⱼ)

    where:
        - 𝑋̄ⱼ is the average value of child 𝑗 (exploitation term).
        - 𝑁ᵢ is the visit count of the parent node 𝑖.
        - 𝑛ⱼ is the visit count of child 𝑗.
        - 𝑐 is the exploration weight.

    Args:
        node (MCTSNode): The parent node.
        exploration_weight (float): Weight for the exploration term in UCB.

    Returns:
        MCTSNode: The selected child node.
    """
    best_child = None
    best_uct_value = -float("inf")

    for child in node.children:
        if child.visits == 0:
            # Encourage exploring unvisited children.
            return child

        # UCB formula.
        exploitation = child.value
        exploration = exploration_weight * math.sqrt(
            2.0 * math.log(node.visits) / child.visits
        )
        ucb_value = exploitation + exploration

        if ucb_value > best_uct_value:
            best_uct_value = ucb_value
            best_child = child

    return best_child


def compute_reward(node: MCTSNode) -> float:
    """
    Compute the reward for a node. Lower cost is better, so we negate the cost.

    Args:
        node (MCTSNode): The node to compute reward for.

    Returns:
        float: The reward value (higher is better).
    """
    # Negate the cost to make it a reward.
    cost = node.cost
    return -cost


def backpropagate(node: MCTSNode, value: float) -> None:
    """
    Backpropagate the value up the tree.

    Args:
        node (MCTSNode): The starting node for backpropagation.
        value (float): The value to backpropagate.
    """
    current = node
    while current is not None:
        current.visits += 1
        current.total_value += value
        current = current.parent


def find_best_node(root: MCTSNode) -> MCTSNode:
    """
    Find the node with the lowest cost in the tree.

    Args:
        root (MCTSNode): The root node of the tree.

    Returns:
        MCTSNode: The node with the lowest cost.
    """

    def traverse(
        node: MCTSNode, best_node: MCTSNode, best_cost: float
    ) -> tuple[MCTSNode, float]:
        # Check if this node has a lower cost.
        if node.cost < best_cost:
            best_node = node
            best_cost = node.cost

        # Recursively check all children.
        for child in node.children:
            child_best, child_cost = traverse(child, best_node, best_cost)
            if child_cost < best_cost:
                best_node = child_best
                best_cost = child_cost

        return best_node, best_cost

    return traverse(root, root, root.cost)[0]


def find_best_nodes(root: MCTSNode) -> list[MCTSNode]:
    """
    Find all nodes with the lowest cost in the tree.

    Args:
        root (MCTSNode): The root node of the tree.

    Returns:
        list[MCTSNode]: All nodes with the lowest cost.
    """

    # Find the minimum cost in the tree.
    def find_min_cost(node: MCTSNode) -> float:
        min_cost = node.cost
        for child in node.children:
            min_cost = min(min_cost, find_min_cost(child))
        return min_cost

    best_cost = find_min_cost(root)

    # Collect all nodes with the minimum cost.
    best_nodes = []

    def collect_best_nodes(node: MCTSNode) -> None:
        if node.cost == best_cost:
            best_nodes.append(node)
        for child in node.children:
            collect_best_nodes(child)

    collect_best_nodes(root)
    return best_nodes


def remove_duplicate_nodes(nodes: list[MCTSNode]) -> list[MCTSNode]:
    """
    Remove duplicate nodes from the list.
    """
    unique_nodes = []
    seen_nodes = set()
    for node in nodes:
        # Create a copy to prevent modifying the original node.
        node_copy = node.scene.clone()

        # Set all masked values to 0.0.
        node_copy[node.inpainting_mask] = 0.0

        node_hash = hash(node_copy.detach().cpu().numpy().tobytes())
        if node_hash not in seen_nodes:
            unique_nodes.append(node)
            seen_nodes.add(node_hash)

    return unique_nodes
