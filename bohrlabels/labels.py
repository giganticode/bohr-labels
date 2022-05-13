# This is automatically generated code. Do not edit manually.

from enum import auto

from bohrlabels.core import Label


class CommitLabel(Label):
    MinorBugFix = auto()
    MajorBugFix = auto()
    CriticalBugFix = auto()
    OtherSeverityLevelBugFix = auto()
    BugFix = MinorBugFix | MajorBugFix | CriticalBugFix | OtherSeverityLevelBugFix
    DocAdd = auto()
    DocSpellingFix = auto()
    DocChange = DocAdd | DocSpellingFix
    TestAdd = auto()
    TestFix = auto()
    TestChange = TestAdd | TestFix
    Reformatting = auto()
    OtherRefactoring = auto()
    Refactoring = Reformatting | OtherRefactoring
    CopyChangeAdd = auto()
    Feature = auto()
    InitialCommit = auto()
    ProjectVersionBump = auto()
    DependencyVersionBump = auto()
    VersionBump = ProjectVersionBump | DependencyVersionBump
    Merge = auto()
    NonBugFix = DocChange | TestChange | Refactoring | CopyChangeAdd | Feature | InitialCommit | VersionBump | Merge
    CommitLabel = BugFix | NonBugFix

    def parent(self):
        return None


class CommitType(Label):
    ConcurrencyBugFix = auto()
    OtherBugFix = auto()
    CommitType = ConcurrencyBugFix | OtherBugFix

    def parent(self):
        return CommitLabel.BugFix


class SStuB(Label):
    WrongIdentifier = auto()
    WrongNumericLiteral = auto()
    WrongModifier = auto()
    WrongBooleanLiteral = auto()
    WrongFunctionName = auto()
    TooFewArguments = auto()
    TooManyArguments = auto()
    WrongFunction = WrongFunctionName | TooFewArguments | TooManyArguments
    WrongBinaryOperator = auto()
    WrongUnaryOperator = auto()
    WrongOperator = WrongBinaryOperator | WrongUnaryOperator
    MissingThrowsException = auto()
    SStuB = WrongIdentifier | WrongNumericLiteral | WrongModifier | WrongBooleanLiteral | WrongFunction | WrongOperator | MissingThrowsException

    def parent(self):
        return CommitType.CommitType


class CommitTangling(Label):
    Tangled = auto()
    NonTangled = auto()
    CommitTangling = Tangled | NonTangled

    def parent(self):
        return CommitLabel.CommitLabel


class SnippetLabel(Label):
    LongMethod = auto()
    LongParameterList = auto()
    Smelly = LongMethod | LongParameterList
    NonSmelly = auto()
    SnippetLabel = Smelly | NonSmelly

    def parent(self):
        return None
