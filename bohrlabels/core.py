from abc import abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from functools import reduce
from typing import List, Optional, Set, Tuple, Type, TypeVar, Union


class Label(int, Flag):
    def __or__(self, other: Union["LabelSet", "Label"]) -> "LabelSet":
        return LabelSet.of(self) | other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def hierarchy_root(cls: Type["Label"]) -> "LabelSubsetBitmap":
        return LabelSubsetBitmap(
            reduce(lambda x, y: x | y, [c.value for c in cls]), cls
        )

    def is_ancestor_of(self, child: "LabelSubclass") -> bool:
        if child is None:
            return False
        if type(self) == type(child):
            return (
                self.value | child.value == self.value
                and self.value & child.value == child.value
            )

        if not hasattr(child, "parent"):
            raise ValueError(
                "Incorrectly defined class. All classes inherited from label must have method 'parent()'"
            )

        return self.is_ancestor_of(child.parent())

    @abstractmethod
    def parent(self):
        pass


LabelSubclass = TypeVar("LabelSubclass", bound=Label)


@dataclass(frozen=True)
class LabelSubsetBitmap:
    label: int
    hierarchy_class: Type[LabelSubclass]

    @classmethod
    def of(cls, label_to_add: LabelSubclass) -> "LabelSubsetBitmap":
        """
        >>> from bohrlabels.labels import CommitLabel
        >>> LabelSubsetBitmap.of(CommitLabel.InitialCommit)
        ['InitialCommit']
        """
        return LabelSubsetBitmap(label_to_add.value, type(label_to_add))

    def __or__(self, other: "LabelSubsetBitmap") -> "LabelSubsetBitmap":
        if not isinstance(other, LabelSubsetBitmap):
            raise ValueError(f"Cannot | LabelSubsetBitmap to {type(other)}")
        if self.hierarchy_class != other.hierarchy_class:
            raise ValueError(
                f"Cannot | LabelSubsetBitmap from different hierarchy classes"
            )

        return LabelSubsetBitmap(self.label | other.label, self.hierarchy_class)

    def __and__(self, other: "LabelSubsetBitmap") -> "LabelSubsetBitmap":
        if not isinstance(other, LabelSubsetBitmap):
            raise ValueError(f"Cannot & LabelSubsetBitmap to {type(other)}")
        if self.hierarchy_class != other.hierarchy_class:
            raise ValueError(
                f"Cannot & LabelSubsetBitmap from different hierarchy classes"
            )

        return LabelSubsetBitmap(self.label & other.label, self.hierarchy_class)

    def __invert__(self):
        return LabelSubsetBitmap(
            ~self.label & self.hierarchy_root().label, self.hierarchy_class
        )

    def is_same_hierarchy(self, other: "LabelSubsetBitmap") -> bool:
        if self.hierarchy_class == other.hierarchy_class:
            return True
        common_hierarchy_class = most_specific_common_hierarchy_class(self, other)
        if (
            self.hierarchy_class != common_hierarchy_class
            and other.hierarchy_class != common_hierarchy_class
        ):
            return True
        if common_hierarchy_class == self.hierarchy_class:
            ancestor, descendent = self, other
        else:
            ancestor, descendent = other, self
        while ancestor.hierarchy_class != descendent.hierarchy_class:
            descendent = descendent.parent()
        return (descendent.label & ancestor.label == descendent.label) or (
            descendent.label & ancestor.label == 0
        )

    def is_ancestor_of(self, child: "LabelSubsetBitmap") -> bool:
        if isinstance(child, Label):
            child = LabelSubsetBitmap.of(child)
        if child is None:
            return False
        if child.hierarchy_class == self.hierarchy_class:
            return (
                self.label | child.label == self.label
                and self.label & child.label == child.label
            )

        return self.is_ancestor_of(child.parent())

    def parent(self) -> Optional["LabelSubsetBitmap"]:
        label = next(iter(self.hierarchy_class))
        if not hasattr(label, "parent"):
            raise ValueError(
                "Incorrectly defined class. All classes inherited from label must have method 'parent()'"
            )
        l = label.parent()
        return LabelSubsetBitmap.of(l) if l is not None else None

    def hierarchy_root(self) -> "LabelSubsetBitmap":
        return self.hierarchy_class.hierarchy_root()

    def to_set_of_labels(self) -> List[LabelSubclass]:  # TODO has to return set
        """
        >>> from bohrlabels.labels import CommitLabel

        # >>> LabelSubsetBitmap(~CommitLabel.Refactoring, hierarchy=CommitLabel).to_commit_labels_set()
        # ['BugFix', 'CiMaintenance', 'CopyChange', 'DependencyChange', 'DocChange', 'Feature', 'GitignoreChange', 'InitialCommit', 'KeybaseChange', 'Merge', 'MetadataChange', 'TestChange', 'VersionBump']
        >>> LabelSubsetBitmap.of(CommitLabel.Refactoring).to_set_of_labels()
        ['Refactoring']
        >>> LabelSubsetBitmap.of(CommitLabel.CommitLabel).to_set_of_labels()
        ['CommitLabel']

        # >>> LabelSubsetBitmap(CommitLabel.Refactoring|CommitLabel.Merge, hierarchy=CommitLabel).to_commit_labels_set()
        # ['Merge', 'Refactoring']
        # >>> LabelSubsetBitmap(CommitLabel.Refactoring|CommitLabel.NonBugFix, hierarchy=CommitLabel).to_commit_labels_set()
        # ['NonBugFix']
        """
        s = set()
        for to_add in self.hierarchy_class:
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

    def __repr__(self):
        return str(self.to_set_of_labels())


def most_specific_common_hierarchy_class(
    first: "LabelSubsetBitmap", second: "LabelSubsetBitmap"
) -> LabelSubclass:
    hieararchies_till_root1, hieararchies_till_root2 = [], []
    while first is not None:
        hieararchies_till_root1.append(first.hierarchy_class)
        first = first.parent()
    while second is not None:
        hieararchies_till_root2.append(second.hierarchy_class)
        second = second.parent()
    result = hieararchies_till_root1[-1]
    ind = -2
    while True:
        try:
            if hieararchies_till_root1[ind] == hieararchies_till_root2[ind]:
                result = hieararchies_till_root1[ind]
                ind -= 1
            else:
                break
        except IndexError:
            break
    return result


@dataclass(frozen=True)
class LabelSet:
    """
    >>> class CommitLabel(Label):
    ...    FEATURE = auto()
    ...    MOVE_CLASS = auto()
    ...    EXCTRACT_METHOD = auto()
    ...    REFACTORING = MOVE_CLASS | EXCTRACT_METHOD
    ...    CONCURRENCY_BUGFIX = auto()
    ...    UI_BUGFIX = auto()
    ...    BUGFIX = CONCURRENCY_BUGFIX | UI_BUGFIX
    ...    COMMIT_LABEL = BUGFIX | FEATURE | REFACTORING
    ...
    ...    def parent(cls):
    ...        return None
    ...
    ...    def __invert__(self):
    ...        return ~LabelSet.of(self)

    >>> class BugFixBySeveriry(Label):
    ...     MAJOR_BUGFIX = auto()
    ...     MINOR_BUGFIX = auto()
    ...     BUGFIX_BY_SEVERITY = MINOR_BUGFIX | MAJOR_BUGFIX
    ...
    ...     def parent(cls):
    ...         return CommitLabel.BUGFIX

    >>> class RefactoringBySize(Label):
    ...    MINOR_REFACTORING = auto()
    ...    MAJOR_REFATORING = auto()
    ...    REFACTORING_BY_SIZE = MINOR_REFACTORING | MAJOR_REFATORING
    ...
    ...    def parent(cls):
    ...        return CommitLabel.REFACTORING

    # TODO draw a hierarchy of classes below here

    >>> RefactoringBySize.MINOR_REFACTORING.is_ancestor_of(CommitLabel.REFACTORING)
    False
    >>> CommitLabel.BUGFIX.is_ancestor_of(BugFixBySeveriry.MINOR_BUGFIX)
    True
    >>> CommitLabel.COMMIT_LABEL.is_ancestor_of(RefactoringBySize.MAJOR_REFATORING)
    True
    >>> CommitLabel.FEATURE.is_ancestor_of(RefactoringBySize.MAJOR_REFATORING)
    False
    >>> CommitLabel.FEATURE.is_ancestor_of(CommitLabel.FEATURE)
    True
    >>> RefactoringBySize.MINOR_REFACTORING.is_ancestor_of(BugFixBySeveriry.MINOR_BUGFIX)
    False

    >>> import sys
    >>> sys.modules[__name__].__dict__.update({'CommitLabel': CommitLabel, 'BugFixBySeveriry': BugFixBySeveriry, 'RefactoringBySize': RefactoringBySize})

    >>> label_set = LabelSet.of(CommitLabel.BUGFIX, BugFixBySeveriry.MINOR_BUGFIX, RefactoringBySize.MINOR_REFACTORING)
    >>> label_set
    ['BUGFIX', 'MINOR_REFACTORING']
    >>> label_set | RefactoringBySize.MAJOR_REFATORING
    ['BUGFIX', 'REFACTORING_BY_SIZE']
    >>> label_set | CommitLabel.CONCURRENCY_BUGFIX
    ['BUGFIX', 'MINOR_REFACTORING']
    >>> label_set | CommitLabel.COMMIT_LABEL
    ['COMMIT_LABEL']
    >>> label_set | LabelSet.of(RefactoringBySize.MINOR_REFACTORING)
    ['BUGFIX', 'MINOR_REFACTORING']
    >>> label_set | LabelSet.of(CommitLabel.CONCURRENCY_BUGFIX)
    ['BUGFIX', 'MINOR_REFACTORING']

    >>> most_specific_common_hierarchy_class(LabelSubsetBitmap.of(CommitLabel.COMMIT_LABEL), LabelSubsetBitmap.of(RefactoringBySize.REFACTORING_BY_SIZE))
    <enum 'CommitLabel'>
    >>> most_specific_common_hierarchy_class(LabelSubsetBitmap.of(RefactoringBySize.REFACTORING_BY_SIZE), LabelSubsetBitmap.of(RefactoringBySize.REFACTORING_BY_SIZE))
    <enum 'RefactoringBySize'>
    >>> most_specific_common_hierarchy_class(LabelSubsetBitmap.of(BugFixBySeveriry.BUGFIX_BY_SEVERITY), LabelSubsetBitmap.of(RefactoringBySize.REFACTORING_BY_SIZE))
    <enum 'CommitLabel'>

    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(CommitLabel.UI_BUGFIX))
    False
    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(CommitLabel.BUGFIX))
    True
    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(CommitLabel.COMMIT_LABEL))
    True
    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(CommitLabel.REFACTORING))
    True
    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(RefactoringBySize.MINOR_REFACTORING))
    True
    >>> LabelSubsetBitmap.of(BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy(LabelSubsetBitmap.of(BugFixBySeveriry.BUGFIX_BY_SEVERITY))
    True

    >>> ~LabelSet.of(CommitLabel.BUGFIX)
    ['FEATURE', 'REFACTORING']
    >>> ~LabelSet.of(CommitLabel.UI_BUGFIX)
    ['CONCURRENCY_BUGFIX', 'FEATURE', 'REFACTORING']
    >>> ~LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX)
    ['FEATURE', 'MAJOR_BUGFIX', 'REFACTORING']
    >>> LabelSet.of(CommitLabel.BUGFIX).belongs_to([CommitLabel.FEATURE, CommitLabel.REFACTORING, BugFixBySeveriry.MINOR_BUGFIX]) is None
    True
    >>> LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING), LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX)])
    ['MINOR_BUGFIX']
    >>> LabelSet.of(RefactoringBySize.MINOR_REFACTORING).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)])
    ['REFACTORING']
    >>> LabelSet.of(CommitLabel.EXCTRACT_METHOD, RefactoringBySize.MINOR_REFACTORING).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)])
    ['REFACTORING']
    >>> LabelSet.of(CommitLabel.FEATURE).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)])
    ['FEATURE']
    >>> LabelSet.of(CommitLabel.COMMIT_LABEL).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)]) is None
    True
    >>> LabelSet.of(CommitLabel.BUGFIX).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)]) is None
    True
    >>> LabelSet.of(CommitLabel.REFACTORING).belongs_to([LabelSet.of(CommitLabel.UI_BUGFIX), LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX)])
    Traceback (most recent call last):
    ...
    ValueError: Categories have to be from the same hierarchy.

    #>>> LabelSet.of(CommitLabel.UI_BUGFIX, BugFixBySeveriry.MAJOR_BUGFIX).belongs_to([LabelSet.of(BugFixBySeveriry.MAJOR_BUGFIX), LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX)])
    #['MAJOR_BUGFIX'] # TODO projection! for the case when heuristic can return not mutually exclusive labels
    >>> LabelSet.of(CommitLabel.BUGFIX).is_same_hierarchy()
    True
    >>> LabelSet.of(CommitLabel.BUGFIX, CommitLabel.FEATURE).is_same_hierarchy()
    True
    >>> LabelSet.of(CommitLabel.BUGFIX, CommitLabel.FEATURE, RefactoringBySize.MINOR_REFACTORING).is_same_hierarchy()
    True
    >>> LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX, CommitLabel.FEATURE, RefactoringBySize.MINOR_REFACTORING).is_same_hierarchy()
    True
    >>> LabelSet.of(CommitLabel.UI_BUGFIX, BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy()
    False
    >>> LabelSet.of(CommitLabel.BUGFIX, CommitLabel.UI_BUGFIX, BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy()
    True
    >>> LabelSet.of(CommitLabel.FEATURE, CommitLabel.UI_BUGFIX, BugFixBySeveriry.MINOR_BUGFIX).is_same_hierarchy()
    False
    >>> LabelSet.of(CommitLabel.BUGFIX, CommitLabel.MOVE_CLASS).belongs_to([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)]) is None
    True
    >>> are_mutually_exclusive([LabelSet.of(CommitLabel.FEATURE), LabelSet.of(CommitLabel.REFACTORING)])
    True
    >>> are_mutually_exclusive([LabelSet.of(CommitLabel.UI_BUGFIX), LabelSet.of(BugFixBySeveriry.MINOR_BUGFIX)])
    False
    """

    labels: Set[LabelSubsetBitmap]

    def __post_init__(self):
        if not isinstance(self.labels, frozenset):
            raise ValueError(f"Labels should be a frozenset but is {type(self.labels)}")

        for label in self.labels:
            if not isinstance(label, LabelSubsetBitmap):
                raise AssertionError()

    @classmethod
    def of(cls, *labels: Union[Label, "LabelSet"]) -> "LabelSet":
        res = set()
        for label_ in labels:
            for label in label_.labels if isinstance(label_, cls) else [label_]:
                res = LabelSet._add_label(
                    res,
                    label
                    if isinstance(label, LabelSubsetBitmap)
                    else LabelSubsetBitmap.of(label),
                )
        return cls(frozenset(res))

    def to_set_of_labels(self) -> List[LabelSubclass]:
        return [l for label in self.labels for l in label.to_set_of_labels()]

    def __repr__(self) -> str:
        sorted_labels = sorted(self.to_set_of_labels())
        return str(sorted_labels)

    @staticmethod
    def _add_two_bitmaps(
        first: LabelSubsetBitmap, second: LabelSubsetBitmap
    ) -> Tuple[LabelSubsetBitmap, Optional[LabelSubsetBitmap]]:
        if not first.is_same_hierarchy(second):
            return first, second

        if first.hierarchy_class == second.hierarchy_class:
            return second | first, None

        common = most_specific_common_hierarchy_class(first, second)
        if common != first.hierarchy_class and common != second.hierarchy_class:
            return first, second

        if common == first.hierarchy_class:
            ancestor, descendent = first, second
        else:
            descendent, ancestor = first, second

        while ancestor.hierarchy_class != descendent.hierarchy_class:
            descendent = descendent.parent()
        if ancestor.label | descendent.label == ancestor.label:
            return ancestor, None
        return first, second

    @staticmethod
    def _add_label(
        labels: Set[LabelSubsetBitmap], label_to_add: LabelSubsetBitmap
    ) -> Set[LabelSubsetBitmap]:
        new_set = set()
        current = label_to_add
        for label in labels:
            current, old = LabelSet._add_two_bitmaps(current, label)
            if old is not None:
                new_set.add(old)
        new_set.add(current)
        return new_set

    def __or__(self, other: Union["LabelSet", Label]) -> "LabelSet":
        label_set = other if isinstance(other, LabelSet) else LabelSet.of(other)

        new_label_set = set(self.labels)
        for label in label_set.labels:
            new_label_set = LabelSet._add_label(new_label_set, label)

        return LabelSet(frozenset(new_label_set))

    def __invert__(self) -> "LabelSet":
        if len(self.labels) > 1:
            raise NotImplemented()

        label = next(iter(self.labels))
        inverse = LabelSet.from_b(~label)
        parent = label.parent()
        if parent is not None:
            inverse |= LabelSet.from_b(~parent)
        return inverse

    def is_same_hierarchy(self):  # TODO optimize
        for label in self.labels:
            for label2 in self.labels:
                if not label.is_same_hierarchy(label2):
                    return False
        return True

    def belongs_to(self, categories: List["LabelSet"]) -> Optional["LabelSet"]:
        if not categories:
            raise ValueError("List of categories cannot be empty")

        category_union: LabelSet = reduce(lambda x, y: x | y, categories)
        if not category_union.is_same_hierarchy():
            raise ValueError("Categories have to be from the same hierarchy.")

        for category in categories:
            if category == category | self:
                return category
        return None

    @classmethod
    def from_bitmap(cls, value: int, hierarchy: Type[LabelSubclass]) -> "LabelSet":
        return LabelSet(frozenset({LabelSubsetBitmap(value, hierarchy)}))

    @classmethod
    def from_b(cls, val):  # TODO check which factory methods are really needed
        return cls.from_bitmap(val.label, val.hierarchy_class)


def are_mutually_exclusive(label_sets: List[LabelSet]) -> bool:
    return reduce(lambda x, y: x | y, label_sets).is_same_hierarchy()


OneOrManyLabels = Union[Label, LabelSet]
