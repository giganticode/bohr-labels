import logging
import re
from glob import glob
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

FlattenedMultiHierarchy = Dict[str, List[List[str]]]

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

bohr_labels_root = Path(__file__).parent.parent


@dataclass
class ParentHierarchy:
    name: str
    mount_point: str


FlattenedNodes = List[Tuple[str, List[str]]]


@dataclass
class FlattenedHierarchy:
    name: str
    parent_hierarchy: Optional[ParentHierarchy]
    nodes: FlattenedNodes = field(default_factory=list)


@dataclass
class LabelHierarchy:
    """
    >>> commit = LabelHierarchy.create_root("Commit")
    >>> bug_fix, refactoring, feature = commit.add_children(["BugFix", "Refactoring", "Feature"])
    >>> minor_refactoring, major_refactoring = refactoring.add_children(["MinorRefactoring", "MajorRefactoring"])
    >>> refactoring.mounted_hierarchy = LabelHierarchy.create_root("RefactoringType")
    >>> moving, renaming = refactoring.mounted_hierarchy.add_children(["Moving", "Renaming"])
    >>> minor, major, critical, other = bug_fix.add_children(["Minor", "Major", "Critical", "OtherSeverityLevel"])
    >>> commit.mounted_hierarchy = LabelHierarchy.create_root("CommitTangling")
    >>> tangled, non_tangled = commit.mounted_hierarchy.add_children(["Tangled", "NonTangled"])

    >>> from pprint import pprint
    >>> pprint(commit.flatten())
    [FlattenedHierarchy(name='Commit', parent_hierarchy=None, nodes=[('Minor', []), ('Major', []), ('Critical', []), ('OtherSeverityLevel', []), ('BugFix', ['Minor', 'Major', 'Critical', 'OtherSeverityLevel']), ('MinorRefactoring', []), ('MajorRefactoring', []), ('Refactoring', ['MinorRefactoring', 'MajorRefactoring']), ('Feature', []), ('Commit', ['BugFix', 'Refactoring', 'Feature'])]),
     FlattenedHierarchy(name='RefactoringType', parent_hierarchy=ParentHierarchy(name='Commit', mount_point='Refactoring'), nodes=[('Moving', []), ('Renaming', []), ('RefactoringType', ['Moving', 'Renaming'])]),
     FlattenedHierarchy(name='CommitTangling', parent_hierarchy=ParentHierarchy(name='Commit', mount_point='Commit'), nodes=[('Tangled', []), ('NonTangled', []), ('CommitTangling', ['Tangled', 'NonTangled'])])]
    """

    label: str
    parent: Optional["LabelHierarchy"]
    children: List["LabelHierarchy"]
    mounted_hierarchy: Optional["LabelHierarchy"] = None

    def __repr__(self) -> str:
        res = self.label
        if self.children:
            res += (
                "{"
                + "|".join(map(lambda x: str(x), self.children))
                + "}"
                + f"-> {self.mounted_hierarchy}"
            )
        return res

    @classmethod
    def create_root(cls, label: str) -> "LabelHierarchy":
        return cls(label, None, [])

    def add_children(self, children: List[str]) -> List["LabelHierarchy"]:
        self.children = [LabelHierarchy(child, self, []) for child in children]
        return self.children

    def flatten(
        self, parent: Optional[ParentHierarchy] = None
    ) -> List[FlattenedHierarchy]:
        hirarchy_name = f"{self.label}"
        hierarchy_tail, other_hierarchies = self._flatten(hirarchy_name)
        return [
            FlattenedHierarchy(hirarchy_name, parent, hierarchy_tail)
        ] + other_hierarchies

    def _flatten(
        self, hierarchy_top: Optional["str"] = None
    ) -> Tuple[FlattenedNodes, List[FlattenedHierarchy]]:
        other_hierarchies: List[FlattenedHierarchy] = []
        main_hierarchy_nodes: FlattenedNodes = []
        for child in self.children:
            child_nodes, lst = child._flatten(hierarchy_top)
            other_hierarchies.extend(lst)
            main_hierarchy_nodes.extend(child_nodes)
        main_hierarchy_nodes.append(
            (self.label, list(map(lambda c: c.label, self.children)))
        )
        if self.mounted_hierarchy:
            other_hierarchies += self.mounted_hierarchy.flatten(
                ParentHierarchy(hierarchy_top, self.label)
            )
        return main_hierarchy_nodes, other_hierarchies


def load(f: List[str]) -> FlattenedMultiHierarchy:
    """
    >>> load([])
    {}
    >>> load(["Commit: BugFix, NonBugFix"])
    {'Commit': [['BugFix', 'NonBugFix']]}
    >>> load(["Commit: BugFix, NonBugFix", "Commit: Tangled, NonTangled"])
    Traceback (most recent call last):
    ...
    ValueError: Parent Commit has at least two hierarchies without having classification type specified
    >>> load(["Commit: BugFix, NonBugFix", "Commit: Tangled"])
    Traceback (most recent call last):
    ...
    ValueError: Commit has to have more than one child.
    >>> load(["Commit: BugFix, NonBugFix", "Commit(CommitTangling): Tangled, NonTangled"])
    {'Commit': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']]}
    >>> load(["Commit: BugFix, NonBugFix", "Commit(CommitTangling): Tangled, NonTangled", "BugFix:Minor,Major"])
    {'Commit': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']], 'BugFix': [['Minor', 'Major']]}
    >>> load(["BugFix, NonBugFix"])
    Traceback (most recent call last):
    ...
    ValueError: Invalid line: BugFix, NonBugFix
     The format must be: Parent: child1, child2, ..., childN
    >>> load(["Commit() : BugFix, NonBugFix"])
    Traceback (most recent call last):
    ...
    ValueError: Invalid parent format: Commit() .
    """
    res: FlattenedMultiHierarchy = {}

    def add_parent_and_children(parent, children, res):
        parent = parent.strip()
        if parent not in res:
            res[parent] = []
        elif isinstance(children, list):
            for elm in res[parent]:
                if isinstance(elm, list):
                    raise ValueError(
                        f"Parent {parent} has at least two hierarchies without having classification type specified"
                    )
        res[parent].append(children)

    for line in f:
        spl_line: List[str] = line.strip("\n").split(":")
        if len(spl_line) != 2:
            raise ValueError(
                f"Invalid line: {line}\n The format must be: Parent: child1, child2, ..., childN"
            )
        left, right = spl_line
        split_list = list(map(lambda x: x.strip(), right.split(",")))
        if len(split_list) < 2:
            raise ValueError(f"{left} has to have more than one child.")
        if re.match("^\\w+$", left):
            add_parent_and_children(left, split_list, res)
        else:
            m = re.match("^(\\w+)\((\\w+)\)$", left)
            if m is None:
                raise ValueError(f"Invalid parent format: {left}.")
            add_parent_and_children(m.group(1), m.group(2), res)
            add_parent_and_children(m.group(2), split_list, res)
    return res


def build_label_tree(
    flattened_multi_hierarchy: FlattenedMultiHierarchy, top_label: str = "Label"
) -> LabelHierarchy:
    """
    >>> build_label_tree({})
    Label
    >>> build_label_tree({"Label": [["BugFix", "NonBugFix"]]})
    Label{BugFix|NonBugFix}-> None
    >>> build_label_tree({'Label': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']]})
    Label{BugFix|NonBugFix}-> CommitTangling{Tangled|NonTangled}-> None
    >>> build_label_tree({'Label': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']], 'BugFix': [['Minor']]})
    Label{BugFix{Minor}-> None|NonBugFix}-> CommitTangling{Tangled|NonTangled}-> None
    """
    tree = LabelHierarchy.create_root(top_label)
    pool = [tree]
    while len(pool) > 0:
        node = pool.pop()
        if node.label in flattened_multi_hierarchy:
            children = flattened_multi_hierarchy[node.label]
            children_nodes = node.add_children(children[0])
            pool.extend(children_nodes)
            for other_children in children[1:]:
                node.mounted_hierarchy = LabelHierarchy(other_children, None, [])
                node = node.mounted_hierarchy
                pool.append(node)
    return tree


def load_label_tree(path_to_labels: Path) -> List[LabelHierarchy]:
    top = []
    for label_file in sorted(glob(f"{path_to_labels}/*.txt")):
        with open(label_file, "r") as f:
            top.append(build_label_tree(load(f.readlines()), Path(label_file).stem))
    return top


TEMPLATE = """# This is automatically generated code. Do not edit manually.

from enum import auto

from bohrlabels.core import Label
{% for hierarchy in hierarchies %}

class {{hierarchy.name}}(Label):{% for row in hierarchy.nodes %}
    {{row[0]}} ={% if row[1] %}{% for r in row[1][:-1] %} {{r}} |{% endfor %} {{row[1][-1]}}{% else %} auto(){% endif %}{% endfor %}

    def parent(self):
        return {% if hierarchy.parent_hierarchy %}{{hierarchy.parent_hierarchy.name}}.{{hierarchy.parent_hierarchy.mount_point}}{% else %}None{% endif %}
{% endfor %}
"""


def parse_labels() -> None:
    label_tree_list = load_label_tree(bohr_labels_root / "labels")
    from jinja2 import Environment

    template = Environment().from_string(TEMPLATE)
    s = template.render(
        hierarchies=[l for label_tree in label_tree_list for l in label_tree.flatten()]
    )
    with open(bohr_labels_root / "bohrlabels" / "labels.py", "w") as f:
        f.write(s)


if __name__ == "__main__":
    parse_labels()
