"""Private composite cross-section reduction sub-package (issue #35,
Phase 2).

A WindIO blade is a thin-walled, multi-cell, laminated-composite beam.
Turning that layup into the 1-D distributed beam properties pyBmodes'
:class:`pybmodes.models.RotatingBlade` FEM consumes (mass/length,
EI_flap, EI_edge, GJ, EA, mass moments of inertia, c.g. / shear-centre
/ tension-centre offsets, structural twist) is a classical-lamination-
theory (CLT) thin-wall shear-flow reduction — the NREL *PreComp*
method (Bir 2006, NREL/TP-500-38929).

Layout (each module independently unit-testable, built sub-phase by
sub-phase so the validation ladder gates the next step):

* :mod:`pybmodes.io._precomp.laminate` — material → reduced stiffness
  ``Q``, ply rotation ``Qbar(theta)``, ABD assembly, membrane
  condensation ``Atilde = A - B D^-1 B``. Pure CLT, no geometry / IO.
* :mod:`pybmodes.io._precomp.geometry` *(SP-2)* — airfoil arc
  parameterisation, spanwise blend, chord/twist/offset application,
  region arc-band resolution across both WindIO dialects.
* :mod:`pybmodes.io._precomp.reduction` *(SP-3/4)* — segment assembly,
  EA / EI principal-axis diagonalisation, single- then multi-cell
  Bredt–Batho ``GJ``, centres, mass moments.

This sub-package is **internal** (underscore-prefixed, same contract as
:mod:`pybmodes.io._elastodyn`); the public entry points live in
``pybmodes.io.windio_blade`` (SP-5). Pure ``numpy``; the WindIO YAML
dependency stays behind the optional ``[windio]`` extra.
"""
