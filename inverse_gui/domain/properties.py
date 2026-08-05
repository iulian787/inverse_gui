"""Property tables and physics gating.

Mirrors amit_AI4NS/utils/optimization/constants.py and properties.py. Kept as a
copy rather than an import because the GUI runs in a different Python environment
than the optimizer and must never import torch.

If the upstream tables change, this file is the single place to update.
"""

from __future__ import annotations

from dataclasses import dataclass

# utils/optimization/constants.py:3-10
ALL_SCALAR_PROPS: tuple[str, ...] = (
    'E', 'nu',
    'E_xx', 'E_yy', 'G_xy', 'nu_xy', 'nu_yx',
    'kappa', 'kappa_x', 'kappa_y',
    'alpha', 'alpha_xx', 'alpha_yy', 'alpha_xy',
    'rho',
)

# utils/optimization/constants.py:11-27 -- also the --<prop>_ref defaults.
PROP_DEFAULTS: dict[str, float] = {
    'E': 240000.0, 'nu': 0.22,
    'E_xx': 240000.0, 'E_yy': 240000.0, 'G_xy': 98000.0,
    'nu_xy': 0.22, 'nu_yx': 0.22,
    'kappa': 178.5, 'kappa_x': 178.5, 'kappa_y': 178.5,
    'alpha': 1.3e-5, 'alpha_xx': 1.3e-5, 'alpha_yy': 1.3e-5, 'alpha_xy': 1e-6,
    'rho': 0.5,
}

# utils/optimization/directives.py:1-6. The isotropic key is DELETED and each
# component gets a copy of the directive at HALF weight -- so `--E target X`
# constrains E_xx and E_yy independently, not their mean.
ISOTROPIC_EXPAND: dict[str, tuple[str, str]] = {
    'E':     ('E_xx', 'E_yy'),
    'nu':    ('nu_xy', 'nu_yx'),
    'kappa': ('kappa_x', 'kappa_y'),
    'alpha': ('alpha_xx', 'alpha_yy'),
}


@dataclass(frozen=True)
class Family:
    """One physics family: its checkpoint flag, its isotropic alias, its components."""
    physics: str                 # 'elastic' | 'thermal_conductivity' | 'thermal_expansion'
    label: str
    ckpt_flag: str               # --ckpt_*_fm
    isotropic: tuple[str, ...]   # props shown in isotropic mode
    components: tuple[str, ...]  # props shown in per-component mode
    unaliased: tuple[str, ...]   # shown in BOTH modes (no isotropic alias exists)

    @property
    def all_props(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.isotropic + self.components + self.unaliased))

    def props_for(self, isotropic_mode: bool) -> tuple[str, ...]:
        base = self.isotropic if isotropic_mode else self.components
        return base + self.unaliased


FAMILIES: tuple[Family, ...] = (
    Family(
        physics='elastic', label='Elastic', ckpt_flag='ckpt_elastic_fm',
        isotropic=('E', 'nu'),
        components=('E_xx', 'E_yy', 'nu_xy', 'nu_yx'),
        unaliased=('G_xy',),          # no isotropic alias upstream
    ),
    Family(
        physics='thermal_conductivity', label='Thermal conductivity',
        ckpt_flag='ckpt_thermal_conductivity_fm',
        isotropic=('kappa',),
        components=('kappa_x', 'kappa_y'),
        unaliased=(),
    ),
    Family(
        physics='thermal_expansion', label='Thermal expansion',
        ckpt_flag='ckpt_thermal_expansion_fm',
        isotropic=('alpha',),
        components=('alpha_xx', 'alpha_yy'),
        unaliased=('alpha_xy',),      # excluded from the `alpha` average upstream
    ),
)

FAMILY_BY_PHYSICS: dict[str, Family] = {f.physics: f for f in FAMILIES}

# rho is always available -- it is m.mean(), computed without any model.
# properties.py:71. Note m == 1 is phase A, so rho is phase A's volume fraction.
ALWAYS_AVAILABLE: frozenset[str] = frozenset({'rho'})


def available_props(physics_loaded: set[str] | frozenset[str]) -> frozenset[str]:
    """Properties the optimizer can actually predict, given the loaded checkpoints.

    Mirrors run_pareto_epsilon_fm_multi_ac.py:1015-1022. The Pareto script warns and
    drops directives outside this set; the single-point script does NOT gate at all,
    which is why the form must -- see validate.UNGATED_DIRECTIVE.
    """
    out = set(ALWAYS_AVAILABLE)
    for physics in physics_loaded:
        fam = FAMILY_BY_PHYSICS.get(physics)
        if fam:
            out.update(fam.all_props)
    return frozenset(out)


def physics_for_prop(prop: str) -> str | None:
    """Which physics must be loaded for `prop` to be predictable. None => always."""
    if prop in ALWAYS_AVAILABLE:
        return None
    for fam in FAMILIES:
        if prop in fam.all_props:
            return fam.physics
    return None


# Display formatting. alpha values are ~1e-5, so a fixed-point format is useless.
PROP_FORMAT: dict[str, str] = {
    p: ('%.3e' if p.startswith('alpha') else '%.4g')
    for p in ALL_SCALAR_PROPS
}

PROP_UNIT: dict[str, str] = {
    'E': 'MPa', 'E_xx': 'MPa', 'E_yy': 'MPa', 'G_xy': 'MPa',
    'nu': '', 'nu_xy': '', 'nu_yx': '',
    'kappa': 'W/mK', 'kappa_x': 'W/mK', 'kappa_y': 'W/mK',
    'alpha': '1/K', 'alpha_xx': '1/K', 'alpha_yy': '1/K', 'alpha_xy': '1/K',
    'rho': 'vol. frac. of phase A',
}

# Phase-endpoint ranges the surrogates were trained over (run_scripts/inverse_run.sh:4-8).
# An achievable effective property lies roughly between the two phase values the user
# set, so these are soft warnings, not bounds.
TRAINING_RANGE: dict[str, tuple[float, float]] = {
    'E': (70000.0, 410000.0),
    'nu': (0.14, 0.30),
    'kappa': (120.0, 237.0),
    'alpha': (3e-6, 2.3e-5),
}


def training_range_for(prop: str) -> tuple[float, float] | None:
    """Training range for a property, following the isotropic alias where needed."""
    if prop in TRAINING_RANGE:
        return TRAINING_RANGE[prop]
    for iso, comps in ISOTROPIC_EXPAND.items():
        if prop in comps and iso in TRAINING_RANGE:
            return TRAINING_RANGE[iso]
    return None


# ---------------------------------------------------------------- phase properties

@dataclass(frozen=True)
class PhaseFlag:
    """A --<x>_A / --<x>_B style flag and the MOOSE-native key it writes."""
    flag_a: str
    flag_b: str
    moose_key: str
    label: str
    dead: bool = False   # accepted by the CLI but read by no fm_multi physics


# utils/optimization/fm_multi.py:57-69
PHASE_FLAGS: tuple[PhaseFlag, ...] = (
    PhaseFlag('E_A', 'E_B', 'youngs_modulus', "Young's modulus"),
    PhaseFlag('nu_A', 'nu_B', 'poissons_ratio', "Poisson's ratio"),
    PhaseFlag('kappa_A', 'kappa_B', 'conductivity', 'Conductivity'),
    PhaseFlag('alpha_A', 'alpha_B', 'thermal_expansion_coefficient', 'Thermal expansion'),
    # density is in _HARDCODED_FM_DEFAULTS but appears in no bulk_props_keys, so
    # FmMultiAdapter never reads it. The flags are accepted and ignored.
    PhaseFlag('rho_A', 'rho_B', 'density', 'Density', dead=True),
)

# physics_registry.json bulk_props_keys / interface_props_keys, enforced by
# FmMultiAdapter.__init__ (fm_multi.py:165-188) with a KeyError if absent.
REQUIRED_BULK_KEYS: dict[str, tuple[str, ...]] = {
    'elastic': ('youngs_modulus', 'poissons_ratio'),
    'thermal_conductivity': ('conductivity',),
    'thermal_expansion': ('youngs_modulus', 'poissons_ratio',
                          'thermal_expansion_coefficient'),
}
REQUIRED_INTERFACE_KEYS: dict[str, tuple[str, ...]] = {
    'elastic': (),
    'thermal_conductivity': ('interfacial_conductivity',),
    'thermal_expansion': (),
}

# utils/optimization/fm_multi.py:35-53. Used to prefill the form.
# (B-phase density is 3.21 for elastic/TE but 3.2 for TC -- upstream inconsistency,
# reproduced faithfully. It is a dead key anyway.)
HARDCODED_PHASE_DEFAULTS: dict[str, dict[str, dict[str, float]]] = {
    'elastic': {
        'A': {'youngs_modulus': 70000.0, 'poissons_ratio': 0.30, 'density': 2.7},
        'B': {'youngs_modulus': 410000.0, 'poissons_ratio': 0.14, 'density': 3.21},
        'AB': {},
    },
    'thermal_conductivity': {
        'A': {'conductivity': 237.0, 'density': 2.7},
        'B': {'conductivity': 120.0, 'density': 3.2},
        'AB': {'interfacial_conductivity': 1.0e7},
    },
    'thermal_expansion': {
        'A': {'youngs_modulus': 70000.0, 'poissons_ratio': 0.30,
              'thermal_expansion_coefficient': 2.3e-5, 'density': 2.7},
        'B': {'youngs_modulus': 410000.0, 'poissons_ratio': 0.14,
              'thermal_expansion_coefficient': 3.0e-6, 'density': 3.21},
        'AB': {},
    },
}


def default_phase_value(moose_key: str, phase: str) -> float | None:
    """First hardcoded default found for a MOOSE key, across physics."""
    for physics in ('elastic', 'thermal_conductivity', 'thermal_expansion'):
        val = HARDCODED_PHASE_DEFAULTS[physics][phase].get(moose_key)
        if val is not None:
            return val
    return None


def required_phase_flags(physics_loaded: set[str] | frozenset[str]) -> set[str]:
    """MOOSE keys that must be present for the loaded physics (A and B both)."""
    keys: set[str] = set()
    for physics in physics_loaded:
        keys.update(REQUIRED_BULK_KEYS.get(physics, ()))
    return keys


def required_interface_keys(physics_loaded: set[str] | frozenset[str]) -> set[str]:
    keys: set[str] = set()
    for physics in physics_loaded:
        keys.update(REQUIRED_INTERFACE_KEYS.get(physics, ()))
    return keys
