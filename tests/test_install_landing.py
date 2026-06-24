"""
A vector-independent structural guard: every phi-arm install must LAND within its own block, at or before the block's
terminator step. An install whose landing PC exceeds the terminator is enqueued for a PC the block never reaches -- a
non-Ret terminator re-keys it onto the taken successor arm, but the Ret wrap silently drops it, a dead install.

This class of defect is invisible to every value comparison (cosim, the example-reference suite, the schedule-
independent MIR interpreter): a dead install that does not alter an output value passes them all, because it is
output-redundant on the vectors. Only a structural invariant catches it. The check is over the settled LIR, independent
of any input vector, so it holds for the data-dependent branch/loop kernels (uart_rx error frames included) too.
"""

import pytest

import holoso
from holoso import FloatFormat
from holoso._frontend import lower as lower_frontend
from holoso._hir import _if_convert as if_convert_pass
from holoso._hir import optimize
from holoso._lir import build
from holoso._lir._ir import Ret
from holoso._mir import lower as lower_to_mir

from ._examples import SPECS
from ._modelref import default_ops


def _build(spec):  # type: ignore[no-untyped-def]
    return build(lower_to_mir(optimize(lower_frontend(spec.make_kernel())), default_ops(spec.formats[0])), spec.name)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_phi_arm_installs_land_within_their_block(spec) -> None:  # type: ignore[no-untyped-def]
    lir = _build(spec)
    for block in lir.blocks:
        for install in (*block.copies, *block.bool_writes):
            landing = install.landing  # block-local; the same fire + read-first edge the model and emitter commit
            assert landing <= block.term_offset, (
                f"{spec.name} block {block.index}: install of {install.dst} lands at {landing}, past the terminator "
                f"{block.term_offset} -- a dead install the Ret wrap would orphan"
            )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_every_example_has_a_single_ret_boundary(spec) -> None:  # type: ignore[no-untyped-def]
    """Sanity anchor for the structural guard above: each kernel terminates at exactly one Ret block."""
    lir = _build(spec)
    rets = [b for b in lir.blocks if isinstance(b.terminator, Ret)]
    assert len(rets) == 1, f"{spec.name}: expected exactly one Ret block, found {len(rets)}"


@pytest.mark.parametrize("name", ["uart_rx", "uart_tx"])
def test_targets_still_exercise_constant_installs(name: str) -> None:
    """
    uart_rx and uart_tx are kernels behind this work: their boolean live-outs ({b3,b4,b5} <- False, True on a
    parity/frame error) and other arms install literal constants with no source to sample, so they fire inline-class and
    land two cycles earlier than a computed-source copy. Pin that these kernels still emit constant phi-arm installs, so
    a kernel-shape change cannot quietly make the recovered-cycle freezes meaningless. The inline-class timing itself is
    pinned end-to-end -- by those frozen lengths (uart_rx 161, uart_tx 142 in test_latency_freeze), by the
    landing <= terminator structural invariant above, and by RTL cosim -- not by re-deriving the install's own helpers.
    """
    spec = next(s for s in SPECS if s.name == name)
    lir = _build(spec)
    const_installs = [x for b in lir.blocks for x in (*b.bool_writes, *b.copies) if x.is_const]
    assert const_installs, f"{name} no longer emits constant phi-arm installs; the kernel shape changed"


def test_resident_register_source_install_is_inline_class() -> None:
    """
    The generalization beyond literal constants: uart_rx installs the rx INPUT directly (b2 <- rx, b4 <- ~rx). A
    register source resident at block entry has nothing to read-first, so the install is classified inline-class
    (``resident_source``) and fires two cycles earlier than a computed-source copy -- exactly like a constant. Pin that
    such a non-const resident-source install is present and so classified; a build that reverted inputs (or any entry-
    resident value) to copy-class would leave this empty. The recovered cycles are pinned end-to-end by the uart_rx
    freeze (161); here we pin that the input path is what is being exercised.
    """
    lir = _build(next(s for s in SPECS if s.name == "uart_rx"))
    resident_non_const = [
        x for b in lir.blocks for x in (*b.bool_writes, *b.copies) if x.resident_source and not x.is_const
    ]
    assert resident_non_const, "uart_rx lost its non-const resident-source (input) install, or the predicate regressed"


class _HoldOrUpdateBool:
    """
    A boolean state held on one arm and updated on the other: ``out`` takes the STATE READ ``self.s`` when ``c`` is
    false and the input ``a`` when true. No bundled example installs a state read as a phi arm, so this pins the third
    entry-resident source kind (after constants and inputs).
    """

    def __init__(self) -> None:
        self.s = False

    def __call__(self, a: bool, c: bool) -> tuple[bool, bool]:
        out = self.s
        if c:
            out = a
        self.s = a
        return out, self.s


def test_state_read_sourced_install_is_inline_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A phi arm that is a STATE READ is resident at block entry (the slot register holds it from the start), so its tail
    install is inline-class -- the generalization's third source kind. Disable if-conversion so the diamond stays a real
    branch and the hold arm installs ``self.s`` by a pc-gated bool write rather than collapsing to a select. Pin both
    that the install is so classified (a non-const resident-source bool write) and -- the black-box teeth -- that the
    held value is the OLD state across a hold/update sweep, which an early-read or clobbered state-read install would
    corrupt (the model vs a fresh Python reference, schedule-independent).
    """
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)
    ops = default_ops(FloatFormat(6, 18))
    lir = build(lower_to_mir(optimize(lower_frontend(_HoldOrUpdateBool().__call__)), ops), "hold_or_update_bool")
    resident_non_const = [x for b in lir.blocks for x in b.bool_writes if x.resident_source and not x.is_const]
    assert resident_non_const, "the state-read phi arm did not install as a resident-source bool write"

    model = holoso.synthesize(_HoldOrUpdateBool().__call__, ops, name="hold_or_update_bool").numerical_model.elaborate()
    reference = _HoldOrUpdateBool()
    for a, c in [(True, False), (False, False), (True, True), (False, False), (True, False), (False, True)]:
        assert tuple(bool(v) for v in model.run(a, c)) == reference(a, c)
