from abc import abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from functools import reduce
from typing import List, Optional, Set, Type, TypeVar, Union


class Label(int, Flag):
    def __or__(self, other: Union["LabelSet", "Label"]):
        if type(self) == type(other):
            return super().__or__(other)
        else:
            return LabelSet.of(self) | other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def hierarchy_root(cls: Type["Label"]) -> "NumericLabel":
        return NumericLabel(reduce(lambda x, y: x | y, cls), cls)

    def is_ancestor_of(self, child: "Label") -> bool:
        if child is None:
            return False
        if type(self) == type(child):
            return self | child == self and self & child == child

        if not hasattr(child, "parent"):
            raise ValueError(
                "Incorrectly defined class. All classes inherited from label must have method 'parent()'"
            )

        return self.is_ancestor_of(child.parent())

    @abstractmethod
    def parent(self):
        pass

    def to_numeric_label(self) -> "NumericLabel":
        return NumericLabel(self.value, type(self))


LabelSubclass = TypeVar("LabelSubclass", bound=Label)


@dataclass(frozen=True)
class NumericLabel:
    label: int
    hierarchy: Type[LabelSubclass]

    def __or__(self, other: "NumericLabel") -> "NumericLabel":
        if not isinstance(other, NumericLabel):
            raise ValueError(f"Cannot | NumericLabel to {type(other)}")
        if self.hierarchy != other.hierarchy:
            raise ValueError(f"Cannot | numerioc labels from different hieararchies")

        return NumericLabel(self.label | other.label, self.hierarchy)

    def __and__(self, other: "NumericLabel") -> "NumericLabel":
        if not isinstance(other, NumericLabel):
            raise ValueError(f"Cannot & NumericLabel to {type(other)}")
        if self.hierarchy != other.hierarchy:
            raise ValueError(f"Cannot & numerioc labels from different hieararchies")

        return NumericLabel(self.label & other.label, self.hierarchy)

    def is_ancestor_of(self, child: "NumericLabel") -> bool:
        if isinstance(child, Label):
            child = child.to_numeric_label()
        if child is None:
            return False
        if child.hierarchy == self.hierarchy:
            return (
                self.label | child.label == self.label
                and self.label & child.label == child.label
            )

        return self.is_ancestor_of(child.parent())

    def parent(self) -> Optional["NumericLabel"]:
        label = next(iter(self.hierarchy))
        if not hasattr(label, "parent"):
            raise ValueError(
                "Incorrectly defined class. All classes inherited from label must have method 'parent()'"
            )
        l = label.parent()
        return l.to_numeric_label() if l is not None else None

    def hierarchy_root(self) -> "NumericLabel":
        return self.hierarchy.hierarchy_root()

    def to_commit_labels_set(self) -> List[LabelSubclass]:
        """
        >>> from bohrlabels.labels import CommitLabel
        >>> NumericLabel(~CommitLabel.Refactoring, CommitLabel).to_commit_labels_set()
        ['BugFix', 'CopyChangeAdd', 'DocChange', 'Feature', 'InitialCommit', 'Merge', 'TestChange', 'VersionBump']
        >>> NumericLabel(CommitLabel.Refactoring, CommitLabel).to_commit_labels_set()
        ['Refactoring']
        >>> NumericLabel(CommitLabel.CommitLabel, CommitLabel).to_commit_labels_set()
        ['CommitLabel']
        >>> NumericLabel(CommitLabel.Refactoring|CommitLabel.Merge, CommitLabel).to_commit_labels_set()
        ['Merge', 'Refactoring']
        >>> NumericLabel(CommitLabel.Refactoring|CommitLabel.NonBugFix, CommitLabel).to_commit_labels_set()
        ['NonBugFix']
        """
        s = set()
        for to_add in self.hierarchy:
            if self.label & to_add.value and (to_add.value | self.label) == self.label:
                rems = []
                add = True
                for (added_name, added_value) in s:
                    if (added_value | to_add.value) == to_add.value:
                        rems.append((added_name, added_value))
                    elif (added_value & to_add.value) == to_add.value:
                        add = False
                        break
                if add:
                    s.add((to_add.name, to_add.value))
                for rem in rems:
                    s.remove(rem)
        return sorted(map(lambda x: x[0], s))

    def to_numeric_label(self) -> "NumericLabel":
        return self


@dataclass(frozen=True)
class LabelSet:
    """
    >>> class A(Label):
    ...    A3 = auto()
    ...    A41 = auto()
    ...    A42 = auto()
    ...    A4 = A41 | A42
    ...    A21 = auto()
    ...    A22 = auto()
    ...    A2 = A21 | A22
    ...    A0 = A2 | A3 | A4
    ...
    ...    def parent(cls):
    ...        return None

    >>> class B(Label):
    ...     B22 = auto()
    ...     B21 = auto()
    ...     A2 = B21 | B22
    ...
    ...     def parent(cls):
    ...         return A.A2

    >>> class C(Label):
    ...    C41 = auto()
    ...    C42 = auto()
    ...    A4 = C41 | C42
    ...
    ...    def parent(cls):
    ...        return A.A4

    #                             A0 <-A
    #                         /   |     \
    #     A2 <-B -------   [A2]   [A3]      [A4]  ----------  A4 <-C
    #   /  \               / \               /  \            /   \
    # B21  B22           A21  A22         A41   A42       (C41)   C42

    >>> C.C41.is_ancestor_of(A.A4)
    False
    >>> A.A2.is_ancestor_of(B.B21)
    True
    >>> A.A0.is_ancestor_of(C.C42)
    True
    >>> A.A3.is_ancestor_of(C.C42)
    False
    >>> A.A3.is_ancestor_of(A.A3)
    True
    >>> C.C41.is_ancestor_of(B.B21)
    False

    >>> import sys
    >>> sys.modules[__name__].__dict__.update({'A': A, 'B': B, 'C': C})

    >>> label_set = LabelSet.of(A.A2, B.B21)
    >>> label_set
    {NumericLabel(label=24, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>)}
    >>> label_set | C.C41
    {NumericLabel(label=24, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>), NumericLabel(label=1, hierarchy=<enum 'C'>)}
    >>> label_set | A.A21
    {NumericLabel(label=24, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>)}
    >>> label_set | A.A0
    {NumericLabel(label=31, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>)}
    >>> label_set | LabelSet.of(C.C41)
    {NumericLabel(label=24, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>), NumericLabel(label=1, hierarchy=<enum 'C'>)}
    >>> label_set | LabelSet.of(A.A21)
    {NumericLabel(label=24, hierarchy=<enum 'A'>), NumericLabel(label=2, hierarchy=<enum 'B'>)}

    >>> LabelSet.of(A.A2).belongs_to([A.A3, A.A4, B.B21])
    Traceback (most recent call last):
    ...
    ValueError: All categories should be from the same hierarchy. However you have categories from different hierarchies: <enum 'A'> and <enum 'B'>
    >>> LabelSet.of(C.C41).belongs_to([A.A3, A.A4])
    NumericLabel(label=6, hierarchy=<enum 'A'>)
    >>> LabelSet.of(A.A42, C.C41).belongs_to([A.A3, A.A4])
    NumericLabel(label=6, hierarchy=<enum 'A'>)
    >>> LabelSet.of(A.A3).belongs_to([A.A3, A.A4])
    NumericLabel(label=1, hierarchy=<enum 'A'>)
    >>> LabelSet.of(A.A0).belongs_to([A.A3, A.A4])
    NumericLabel(label=31, hierarchy=<enum 'A'>)
    >>> LabelSet.of(A.A2).belongs_to([A.A3, A.A4])
    NumericLabel(label=31, hierarchy=<enum 'A'>)
    >>> LabelSet.of(A.A2, A.A41).belongs_to([A.A3, A.A4])
    NumericLabel(label=31, hierarchy=<enum 'A'>)
    """

    labels: Set[NumericLabel]

    def __post_init__(self):
        if not isinstance(self.labels, frozenset):
            raise ValueError(f"Labels should be a frozenset but is {type(self.labels)}")

        for label in self.labels:
            if not isinstance(label, NumericLabel):
                raise AssertionError()

    @classmethod
    def of(cls, *labels: Union[LabelSubclass, NumericLabel]):
        res = set()
        for label in labels:
            res = LabelSet._add_label(res, label)
        return cls(frozenset(res))

    def __repr__(self) -> str:
        sorted_labels = sorted(self.labels, key=lambda l: l.hierarchy.__name__)
        return "{" + ", ".join(map(lambda l: repr(l), sorted_labels)) + "}"

    @staticmethod
    def _add_label(
        labels: Set[NumericLabel], label_to_add: Union[LabelSubclass, NumericLabel]
    ) -> Set[LabelSubclass]:
        if isinstance(label_to_add, Label):
            label_to_add = label_to_add.to_numeric_label()
        elif isinstance(label_to_add, NumericLabel):
            pass
        else:
            raise AssertionError(label_to_add)
        flag_to_remove = None
        for flag in labels:
            if flag.hierarchy == label_to_add.hierarchy:
                flag_to_remove = flag
                label_to_add = label_to_add | flag
                break
        if flag_to_remove:
            labels.remove(flag_to_remove)
        labels.add(label_to_add)
        return labels

    def __or__(self, other: Union["LabelSet", Label]) -> "LabelSet":
        label_set = other if isinstance(other, LabelSet) else LabelSet.of(other)

        new_label_set = set(self.labels)
        for label in label_set.labels:
            new_label_set = LabelSet._add_label(new_label_set, label)

        return LabelSet(frozenset(new_label_set))

    @staticmethod
    def get_label_union(labels: List[NumericLabel]) -> NumericLabel:
        label_union = labels[0]

        for label in labels:
            if label.hierarchy != labels[0].hierarchy:
                raise ValueError(
                    "All categories should be from the same hierarchy. "
                    f"However you have categories from different hierarchies: {labels[0].hierarchy} and {label.hierarchy}"
                )
            label_union |= label
        return label_union

    def belongs_to(
        self, categories: List[Union[LabelSubclass, NumericLabel]]
    ) -> NumericLabel:
        if not categories:
            raise ValueError("List of categories cannot be empty")
        categories: List[NumericLabel] = list(
            map(
                lambda c: (c.to_numeric_label() if isinstance(c, Label) else c),
                categories,
            )
        )
        category_union: NumericLabel = LabelSet.get_label_union(categories)

        result = category_union.hierarchy_root()
        for label in self.labels:
            if category_union.is_ancestor_of(label):
                while label.hierarchy != category_union.hierarchy:
                    label = label.parent()
                for category in categories:
                    if category.label & label.label:
                        result &= category | label

        return result


def belongs_to(
    label: NumericLabel, categories: List[Union[LabelSubclass, NumericLabel]]
) -> NumericLabel:
    return LabelSet.of(label).belongs_to(categories)


def to_numeric_label(
    label: Union[int, LabelSubclass, NumericLabel], hierarchy: Type[LabelSubclass]
) -> NumericLabel:
    """
    >>> from bohrlabels.labels import CommitLabel, SStuB
    >>> to_numeric_label(1, CommitLabel)
    NumericLabel(label=1, hierarchy=<enum 'CommitLabel'>)
    >>> to_numeric_label(CommitLabel.InitialCommit, CommitLabel)
    NumericLabel(label=CommitLabel.InitialCommit, hierarchy=<enum 'CommitLabel'>)
    >>> to_numeric_label(1., CommitLabel)
    Traceback (most recent call last):
    ...
    ValueError: Invalid label type: <class 'float'>
    """
    if isinstance(label, Label):
        if hierarchy != type(label):
            raise ValueError(
                f"Hierarchy mismatch. Passed label is {label}, passed hierarchy is {hierarchy}"
            )
        return label.to_numeric_label()
    elif isinstance(label, int):
        return NumericLabel(label, hierarchy)
    elif isinstance(label, NumericLabel):
        return label
    else:
        raise ValueError(f"Invalid label type: {type(label)}")


Labels = Union[Label, LabelSet]
