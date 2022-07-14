# This is automatically generated code. Do not edit manually.

from enum import auto

from bohrlabels.core import Label


class CommitLabel(Label):
    ConcurrencyBugFix = auto()
    OtherBugFix = auto()
    BugFix = ConcurrencyBugFix | OtherBugFix
    ChangeLogAdd = auto()
    OtherDocAdd = auto()
    DocAdd = ChangeLogAdd | OtherDocAdd
    DocSpellingFix = auto()
    DocRemove = auto()
    DocFix = auto()
    DocChange = DocAdd | DocSpellingFix | DocRemove | DocFix
    Reformatting = auto()
    RemoveUnused = auto()
    OtherMinorRefactoring = auto()
    MinorRefactoring = Reformatting | RemoveUnused | OtherMinorRefactoring
    NormalRefactoring = auto()
    MajorRefactoring = auto()
    Refactoring = MinorRefactoring | NormalRefactoring | MajorRefactoring
    CopyChange = auto()
    MajorFeature = auto()
    Enhancement = auto()
    Feature = MajorFeature | Enhancement
    InitialCommit = auto()
    ProjectVersionBump = auto()
    DependencyVersionBump = auto()
    DependencyRemove = auto()
    VersionBump = ProjectVersionBump | DependencyVersionBump | DependencyRemove
    DependencyChange = auto()
    CiMaintenance = auto()
    GitignoreChange = auto()
    KeybaseChange = auto()
    TestAdd = auto()
    TestFix = auto()
    TestChange = TestAdd | TestFix
    MetadataChange = auto()
    Merge = auto()
    NonBugFix = DocChange | Refactoring | CopyChange | Feature | InitialCommit | VersionBump | DependencyChange | CiMaintenance | GitignoreChange | KeybaseChange | TestChange | MetadataChange | Merge
    CommitLabel = BugFix | NonBugFix

    def parent(self):
        return None


class BugSeverity(Label):
    MinorBugFix = auto()
    MajorBugFix = auto()
    CriticalBugFix = auto()
    OtherSeverityLevelBugFix = auto()
    BugSeverity = MinorBugFix | MajorBugFix | CriticalBugFix | OtherSeverityLevelBugFix

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
        return BugSeverity.BugSeverity


class CommitTangling(Label):
    Tangled = auto()
    NonTangled = auto()
    CommitTangling = Tangled | NonTangled

    def parent(self):
        return CommitLabel.CommitLabel


class CommitSize(Label):
    Huge = auto()
    Normal = auto()
    OneLine = auto()
    Empty = auto()
    CommitSize = Huge | Normal | OneLine | Empty

    def parent(self):
        return CommitTangling.CommitTangling


class MatchLabel(Label):
    Match = auto()
    NoMatch = auto()
    MatchLabel = Match | NoMatch

    def parent(self):
        return None


class SnippetLabel(Label):
    LongMethod = auto()
    LongParameterList = auto()
    Smelly = LongMethod | LongParameterList
    NonSmelly = auto()
    SnippetLabel = Smelly | NonSmelly

    def parent(self):
        return None
