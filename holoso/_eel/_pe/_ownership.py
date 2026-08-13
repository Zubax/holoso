"""
The mutation/ownership model. The rules below are normative here, alongside the state-install disjointness
rules enforced in `_state`.

Every aggregate value is an allocation with a MONOTONE state: unique, then shared or escaped -- never back.
All events that create a second handle move the state forward:

- binding the same allocation to another name (aliasing, multi-target assignment, caller-argument to
  callee-parameter at inlining, unpacking targets receiving aggregates);
- aggregate-valued reads: an index, slice, iteration item, or unpacked element that IS an aggregate marks
  parent and extracted allocation shared -- except the syntactic prefix of a store-target path (`C[i][j] = s`
  does not share by reading `C[i]` on the way to the store);
- embedding: an aggregate flowing into a literal shares its descendants with the result; sequence slices
  COPY the top level (a fresh allocation) and share extracted aggregate descendants;
- derivation: ANY tensor derived from an existing one (a tensor slice, `.T`, `.flatten()`, `asarray`)
  is shared with its source -- the compiler deliberately models no copy-vs-view distinction. A sequence
  converted to a tensor is a leaf copy, not a derivation: the scalar leaves are values, so no handle relates
  the two;
- escape: returning from the kernel, installing into a state path, or being stored as an element; aggregate
  ROOTS originating outside the kernel's own construction (module globals, closure cells, captured defaults,
  and the kernel's own array parameters) enter ESCAPED -- CPython would make a mutation observable outside
  the kernel, which hardware cannot honor;
- borrow: the iterable of an active loop/comprehension is borrowed for the loop's duration (a scoped
  overlay on the iterated allocation, not a monotone state: the borrow lifts when the loop ends, and items
  extracted during iteration remain shared via the aggregate-valued-read event, which carries the safety).

A while header evaluated speculatively (to see whether the test folds) rolls back its environment effects
when the loop residualizes, but the ownership events it fired persist -- allocation states are global
monotone facts, so speculation is at worst stricter, never wrong.

Store admission is PATH-QUANTIFIED (amended A1): a mutation is admitted iff EVERY allocation along the
store path -- from the root handle through the allocation holding the written scalar leaf -- is unique and
unborrowed at the store step, checked after the RHS and index temps are evaluated. The final holder must be
a tensor (sequences are immutable structure; a sequence may still sit mid-path), the path must end at a
SCALAR slot, and the stored value must be a scalar of the tensor's family (ints promote into a float
family; a float into an integer family truncates on the host and rejects). Soundness: an admitted store mutates
a chain of allocations reachable through exactly one handle at every level, where reference and value
semantics coincide; every second-handle route is one of the events above, so no heap model, view tracking,
or escape analysis is needed. Everything outside that envelope is rejected with rebind advice, never
reinterpreted. Allocation states are global monotone facts, so an event under either arm of a residual
branch conservatively binds the whole join (a share in the then-arm blocks a store in the else-arm) --
per-path sound in both directions, occasionally stricter than one path alone.
"""

from ._values import Allocation, AllocationState, SequenceValue, TensorValue, Value


def share(value: Value) -> None:
    for allocation in allocations(value):
        if allocation.state is AllocationState.UNIQUE:
            allocation.state = AllocationState.SHARED


def escape(value: Value) -> None:
    for allocation in allocations(value):
        allocation.state = AllocationState.ESCAPED


def borrow(value: Value) -> Allocation | None:
    match value:
        case SequenceValue(allocation=allocation) | TensorValue(allocation=allocation):
            allocation.borrows += 1
            return allocation
        case _:
            return None


def release(allocation: Allocation | None) -> None:
    if allocation is not None:
        assert allocation.borrows > 0
        allocation.borrows -= 1


def mutable(allocation: Allocation) -> bool:
    return allocation.state is AllocationState.UNIQUE and allocation.borrows == 0


def blame(allocation: Allocation) -> str:
    if allocation.borrows > 0:
        return "it is being iterated by an enclosing loop"
    match allocation.state:
        case AllocationState.SHARED:
            return "it is shared (reachable through another name or container); rebind a fresh value instead"
        case AllocationState.ESCAPED:
            return (
                "it arrived from outside the kernel (a global, a closure, a default, or a parameter) or has "
                "escaped through a return or a state install, so the change would be observable elsewhere"
            )
        case AllocationState.UNIQUE:
            raise AssertionError("a unique allocation draws no blame")


def allocations(value: Value) -> list[Allocation]:
    match value:
        case SequenceValue(items=items, allocation=allocation):
            found = [allocation]
            for item in items:
                found.extend(allocations(item))
            return found
        case TensorValue(allocation=allocation):
            return [allocation]
        case _:
            return []
