"""K-expression compression via nested-set decomposition.

Approach (Celko nested-set model + FK placeholder):
  1. Decode the chromosome head into a tree.
  2. Annotate every node with (left, right) via one DFS — gives subtree
     size in O(1) thereafter.
  3. Find the largest subtree whose size <= sub_h. Simplify it (sympy ->
     karva via the "visit" function), record the simplified tokens, and
     replace the subtree in the parent tree with a FK placeholder
     terminal.
  4. Re-annotate and repeat until no compressible subtree remains.
  5. Serialise the (now shrunken) parent tree back to karva. Substitute
     FK placeholders with their simplified token lists.

Public API:
  compress_gene(gene, pset, visit_subtree, sub_h=10, max_passes=2) -> (head, tail)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from geppy.core.symbol import Function, Terminal


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A single node in the expression tree. ``tok`` is the original geppy
    Function/Terminal symbol. Internal nodes have ``children``; leaves don't.
    Celko numbers are filled by ``annotate``.

    For FK placeholders, ``tok`` is None and ``fk_id`` is set; the node
    behaves as a terminal at serialise-time.
    """
    tok: Optional[object]
    children: list = field(default_factory=list)
    left: int = 0
    right: int = 0
    size: int = 0  # total nodes in this subtree (including self)
    fk_id: Optional[int] = None  # FK placeholder id, or None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_fk(self) -> bool:
        return self.fk_id is not None


# ---------------------------------------------------------------------------
# Decode head -> tree
# ---------------------------------------------------------------------------

def decode_head_to_tree(head: list, tail: list) -> Node:
    """Decode a gene's level-order head (and tail) into a tree.

    GEP property: walking head left-to-right while tracking arity demand
    tells us where each child lives in the combined head+tail stream.
    Tail tokens are terminals.
    """
    stream = list(head) + list(tail)
    n_head = len(head)
    # Build all nodes for stream positions.
    nodes: list[Node] = []
    for i, tok in enumerate(stream):
        nodes.append(Node(tok=tok))

    next_slot = 1
    for i in range(n_head):
        tok = head[i]
        if isinstance(tok, Function):
            for _ in range(tok.arity):
                nodes[i].children.append(nodes[next_slot])
                next_slot += 1
    return nodes[0]


# ---------------------------------------------------------------------------
# Celko annotation (DFS left/right numbering + subtree size)
# ---------------------------------------------------------------------------

def annotate(root: Node) -> None:
    """Fill .left, .right, .size on every node via one DFS."""
    counter = [0]

    def _visit(n: Node) -> int:
        counter[0] += 1
        n.left = counter[0]
        size = 1
        for c in n.children:
            size += _visit(c)
        counter[0] += 1
        n.right = counter[0]
        n.size = size
        return size

    _visit(root)


# ---------------------------------------------------------------------------
# Find the largest compressible subtree
# ---------------------------------------------------------------------------

def find_largest_compressible(root: Node, sub_h: int,
                                exclude: set | None = None) -> Optional[Node]:
    """Return the deepest, largest subtree with size <= sub_h AND containing
    at least one Function (compressing single terminals is pointless). Root
    is eligible — caller decides whether to special-case it. Already-tried
    nodes (those whose visit returned the same Node back) are skipped via
    the ``exclude`` set to avoid infinite loops.

    Returns None if nothing compressible remains.
    """
    exclude = exclude or set()
    candidates: list[Node] = []

    def _walk(n: Node, depth: int) -> None:
        if id(n) in exclude:
            for c in n.children:
                _walk(c, depth + 1)
            return
        if n.size <= sub_h and _has_function(n):
            candidates.append((n.size, depth, n))
            return  # don't recurse into a candidate's children
        for c in n.children:
            _walk(c, depth + 1)

    _walk(root, 0)
    if not candidates:
        return None
    # Prefer largest size; tiebreak by deepest.
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    return candidates[0][2]


def _has_function(n: Node) -> bool:
    if isinstance(n.tok, Function):
        return True
    for c in n.children:
        if _has_function(c):
            return True
    return False


# ---------------------------------------------------------------------------
# Replace subtree with a FK placeholder
# ---------------------------------------------------------------------------

def replace_subtree(root: Node, target: Node, fk_node: Node) -> bool:
    """In-place: find target in root's tree and swap with fk_node. Returns
    True if found and replaced.
    """
    if root is target:
        # Caller should never ask to replace the root.
        return False
    for i, c in enumerate(root.children):
        if c is target:
            root.children[i] = fk_node
            return True
        if replace_subtree(c, target, fk_node):
            return True
    return False


# ---------------------------------------------------------------------------
# Serialise tree back to head/tail
# ---------------------------------------------------------------------------

def serialise_tree(root: Node) -> tuple[list, list]:
    """Walk the tree in level order, producing (head, tail) token lists.

    Strategy: BFS. At each level, function nodes contribute to the head
    until we've laid out the full tree. Terminals (and FK placeholders)
    are emitted in their slots; they don't expand.

    For GEP-validity, we then pad the tail with the BFS-emitted terminals
    that appeared after the last function node.
    """
    # Level-order BFS
    order: list[Node] = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in n.children:
            queue.append(c)
    # Find the last function-node position in BFS order — that defines
    # where the head ends.
    last_fn = -1
    for i, n in enumerate(order):
        if isinstance(n.tok, Function):
            last_fn = i
    # Head: positions 0..last_fn. Tail: positions last_fn+1..end.
    head_nodes = order[:last_fn + 1] if last_fn >= 0 else order[:1]
    tail_nodes = order[last_fn + 1:] if last_fn >= 0 else order[1:]
    return [n for n in head_nodes], [n for n in tail_nodes]


def _node_to_token(n: Node, fk_placeholder_terminals: dict):
    """Convert a Node back to a geppy token. FK placeholders use cached
    fake-terminal objects keyed by fk_id."""
    if n.is_fk:
        return fk_placeholder_terminals[n.fk_id]
    return n.tok


# ---------------------------------------------------------------------------
# Compression pipeline
# ---------------------------------------------------------------------------

def compress_gene(
    gene,
    pset,
    visit_subtree: Callable,
    sub_h: int = 10,
    max_passes: int = 2,
    rng: random.Random | None = None,
):
    """Compress a gene by extracting compressible subtrees, simplifying
    each, and substituting tokens back. Returns (head_tokens, tail_tokens).

    ``visit_subtree(root_node, pset) -> (head_tokens, tail_tokens) | None``
        Called once per chosen subtree. Should:
          - convert the Node tree (rooted at root_node) to sympy
          - run sympy.simplify
          - emit the simplified expression back as karva tokens
          - return None on any failure (caller falls back to original tokens)
    """
    rng = rng or random.Random()
    head = list(gene.head)
    tail = list(gene.tail)

    for _pass in range(max_passes):
        root = decode_head_to_tree(head, tail)
        annotate(root)
        # If the whole tree fits within sub_h, no decomposition needed.
        sub_genes: list[tuple[list, list]] = []  # (head_tokens, tail_tokens) per FK
        # FK terminals are placeholder objects implementing geppy's Terminal protocol.
        fk_placeholders: dict[int, object] = {}

        # sub_genes is a list of replacement *Node subtrees* keyed by FK id.
        # If visit returned None, the entry is the original subtree (so the
        # FK round-trips to the same Node, semantically unchanged).
        sub_trees: list[Node] = []
        # Track Node identities whose visit returned a result equivalent
        # (identical tree-shape) to the input, so the picker doesn't loop
        # back to them.
        exclude: set[int] = set()
        changed = False
        while True:
            target = find_largest_compressible(root, sub_h, exclude=exclude)
            if target is None:
                break
            simplified_tokens = visit_subtree(target, pset)
            if simplified_tokens is None:
                # Fallback: re-use the original Node subtree unchanged.
                replacement_node = target
                exclude.add(id(target))  # don't pick again
            else:
                fk_h, fk_t = simplified_tokens
                replacement_node = decode_head_to_tree(list(fk_h), list(fk_t))
                # If the simplified result is structurally same size as the
                # original, treat as no-op for exclude purposes — otherwise
                # we'd loop on no-progress simplifications.
                annotate(target)
                annotate(replacement_node)
                if replacement_node.size >= target.size:
                    exclude.add(id(target))
            # Special case: if target IS root, replace root in place
            # (replace_subtree can't substitute the root).
            if target is root:
                root.tok = replacement_node.tok
                root.children = replacement_node.children
                root.fk_id = None
                changed = True
                annotate(root)
                continue
            fk_id = len(sub_trees)
            sub_trees.append(replacement_node)
            fk_node = Node(tok=_FKTerminal(fk_id), fk_id=fk_id)
            ok = replace_subtree(root, target, fk_node)
            assert ok, "subtree not found in parent"
            annotate(root)  # re-number after shrink
            changed = True

        if not changed:
            break

        # Substitute every FK in the parent tree with its replacement Node
        # subtree (either the original or a simplified one).
        _expand_fks_in_tree(root, sub_trees)

        # Now serialise the fully-expanded tree (BFS / level-order) into a
        # single stream. Head = up to and including the last function; tail
        # = everything after, re-padded with random terminals to GEP rule.
        stream = _bfs_stream(root)
        last_fn = -1
        for i, tok in enumerate(stream):
            if isinstance(tok, Function):
                last_fn = i
        if last_fn < 0:
            # No functions left — gene collapsed to a single terminal.
            new_head = [stream[0]]
            new_tail = []
        else:
            new_head = stream[:last_fn + 1]
            new_tail = stream[last_fn + 1:]

        max_arity = getattr(pset, "max_arity", None) or max(
            (p.arity for p in pset.functions), default=2)
        target_tail_len = len(new_head) * (max_arity - 1) + 1
        terminals = list(pset.terminals)
        while len(new_tail) < target_tail_len:
            new_tail.append(rng.choice(terminals))
        new_tail = new_tail[:target_tail_len]

        head, tail = new_head, new_tail

    return head, tail


# ---------------------------------------------------------------------------
# FK placeholder terminal (duck-types as geppy Terminal for serialisation)
# ---------------------------------------------------------------------------

def _expand_fks_in_tree(root: Node, sub_trees: list[Node]) -> None:
    """Walk the tree; whenever a child Node is an FK, replace it with the
    stored replacement subtree (Node). Recursively expand nested FKs.
    """
    def _walk(n: Node) -> None:
        for i, c in enumerate(n.children):
            if c.is_fk:
                sub_root = sub_trees[c.fk_id]
                n.children[i] = sub_root
                _walk(sub_root)
            else:
                _walk(c)
    if root.is_fk:
        sub_root = sub_trees[root.fk_id]
        root.tok = sub_root.tok
        root.children = sub_root.children
        root.fk_id = None
    _walk(root)


def _bfs_stream(root: Node) -> list:
    """Return the tree's nodes' tokens in BFS / level order."""
    out = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        out.append(n.tok)
        for c in n.children:
            queue.append(c)
    return out


class _FKTerminal(Terminal):
    """A throwaway Terminal subclass used only during compression. Holds a
    foreign-key id pointing into the compressor's sub_genes list.

    We never persist these — they're substituted out before returning.
    """
    def __init__(self, fk_id: int):
        super().__init__(name=f"_FK_{fk_id}", value=None)
        self._fk_id = fk_id

    @property
    def fk_id(self) -> int:
        return self._fk_id

    @property
    def arity(self) -> int:
        return 0
