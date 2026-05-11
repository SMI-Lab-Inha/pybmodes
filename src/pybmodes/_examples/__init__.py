"""Bundled example inputs and reference decks.

This sub-package vendors two trees that ship inside the wheel so a
``pip install pybmodes`` user can copy them out to a working directory
without keeping a full git clone of the repository.

- ``sample_inputs/`` — pyBmodes-authored, MIT-licensed ``.bmi`` and
  section-property ``.dat`` files. Four analytical-reference cases
  (uniform blade, tower with top mass, rotating uniform blade,
  pinned-free cable) plus seven reference-wind-turbine sub-cases under
  ``reference_turbines/``. ``verify.py`` runs the four analytical
  cases against closed-form references.
- ``reference_decks/`` — six pre-patched ElastoDyn decks (three
  land/monopile + three floating) whose polynomial blocks have been
  regenerated from the structural inputs via ``Tower.from_elastodyn``.

The trees are intentionally treated as opaque data; the ``.py``
helpers inside them (``verify.py``, ``reference_turbines/build.py``)
are intended to be run *after* vendoring out via
``pybmodes examples --copy <dir>``, not imported.

Users discover the bundles through three paths:

- ``pybmodes examples --copy DIR`` — CLI that copies one or both
  trees into a user-supplied directory.
- Browsing the wheel install directly under
  ``site-packages/pybmodes/_examples/``.
- For developers working from a source checkout, the GitHub source
  tree under the same path.
"""
