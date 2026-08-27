# Task and parameter semantics: Schlueter et al. (2021), Section 2.1 and Table 2.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L315-L321
GTOPX1_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in insertion into an orbit around Saturn. \
The target orbit has a pericenter radius of 108,950 km and an eccentricity of 0.98. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Venus -> Venus -> Earth -> Jupiter -> Saturn (end). \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 6 continuous parameters \
and must satisfy 4 constraints, \
which impose lower limits on the pericenter radii of the 4 fly-by maneuvers.

The 6 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Time interval from departure at Earth to the first Venus fly-by, in days.
- p2: Time interval from the first Venus fly-by to the second Venus fly-by, in days.
- p3: Time interval from the second Venus fly-by to the Earth fly-by, in days.
- p4: Time interval from the Earth fly-by to the Jupiter fly-by, in days.
- p5: Time interval from the Jupiter fly-by to orbit insertion at Saturn, in days.

The parameter bounds are:
- p0: [-1000, 0].
- p1: [30, 400].
- p2: [100, 470].
- p3: [30, 400].
- p4: [400, 2000].
- p5: [1000, 6000].\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.2 and Table 3.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L322-L330
GTOPX2_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in a rendezvous with Saturn. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Venus -> Venus -> Earth -> Jupiter -> Saturn (end). \
The trajectory includes 1 deep-space maneuver (DSM) on each of the 5 transfer legs. \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 22 continuous parameters.

The 22 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Initial hyperbolic excess speed in km/s.
- p2, p3: Normalized polar-coordinate parameters \
that determine the direction of the initial hyperbolic excess velocity.
- p4: Time interval from departure at Earth to the first Venus fly-by, in days.
- p5: Time interval from the first Venus fly-by to the second Venus fly-by, in days.
- p6: Time interval from the second Venus fly-by to the Earth fly-by, in days.
- p7: Time interval from the Earth fly-by to the Jupiter fly-by, in days.
- p8: Time interval from the Jupiter fly-by to rendezvous with Saturn, in days.
- p9-p13: Fraction of the time interval after which the DSM occurs \
in each corresponding transfer leg.
- p14-p17: Fly-by pericenter radius at each fly-by encounter, \
measured in radii of the encountered planet.
- p18-p21: B-plane rotation angles \
that orient the post-fly-by outgoing relative-velocity vector \
about the incoming relative-velocity direction at the 4 fly-by encounters, in radians.

The parameter bounds are:
- p0: [-1000, 0].
- p1: [3, 5].
- p2, p3: [0, 1] each.
- p4: [100, 400].
- p5: [100, 500].
- p6: [30, 300].
- p7: [400, 1600].
- p8: [800, 2200].
- p9-p13: [0.01, 0.9] each.
- p14, p15: [1.05, 6] each.
- p16: [1.15, 6.5].
- p17: [1.7, 291].
- p18-p21: [-\\pi, \\pi] each.\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.3 and Table 4.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L331-L339
GTOPX3_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in a rendezvous with Mercury. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Earth -> Venus -> Venus -> Mercury (end). \
The trajectory includes 1 deep-space maneuver (DSM) on each of the 4 transfer legs. \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 18 continuous parameters.

The 18 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Initial hyperbolic excess speed in km/s.
- p2, p3: Normalized polar-coordinate parameters \
that determine the direction of the initial hyperbolic excess velocity.
- p4: Time interval from departure at Earth to the Earth fly-by, in days.
- p5: Time interval from the Earth fly-by to the first Venus fly-by, in days.
- p6: Time interval from the first Venus fly-by to the second Venus fly-by, in days.
- p7: Time interval from the second Venus fly-by to rendezvous with Mercury, in days.
- p8-p11: Fraction of the time interval after which the DSM occurs \
in each corresponding transfer leg.
- p12-p14: Fly-by pericenter radius at each fly-by encounter, \
measured in radii of the encountered planet.
- p15-p17: B-plane rotation angles \
that orient the post-fly-by outgoing relative-velocity vector \
about the incoming relative-velocity direction at the 3 fly-by encounters, in radians.

The parameter bounds are:
- p0: [1000, 4000].
- p1: [1, 5].
- p2, p3: [0, 1] each.
- p4-p7: [30, 400] each.
- p8-p11: [0.01, 0.99] each.
- p12-p14: [1.1, 6] each.
- p15-p17: [-\\pi, \\pi] each.\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.4 and Table 5.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L340-L350
GTOPX4_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in insertion into an orbit around Mercury. \
The target orbit has a pericenter radius of 2,640 km and an eccentricity of 0.704. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Venus -> Venus -> Mercury -> Mercury -> Mercury -> Mercury (end). \
The mission includes three resonant fly-bys at Mercury before orbit insertion. \
The trajectory includes 1 deep-space maneuver (DSM) on each of the 6 transfer legs. \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 26 continuous parameters.

The 26 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Initial hyperbolic excess speed in km/s.
- p2, p3: Normalized polar-coordinate parameters \
that determine the direction of the initial hyperbolic excess velocity.
- p4: Time interval from departure at Earth to the first Venus fly-by, in days.
- p5: Time interval from the first Venus fly-by to the second Venus fly-by, in days.
- p6: Time interval from the second Venus fly-by to the first Mercury fly-by, in days.
- p7: Time interval from the first Mercury fly-by to the second Mercury fly-by, in days.
- p8: Time interval from the second Mercury fly-by to the third Mercury fly-by, in days.
- p9: Time interval from the third Mercury fly-by \
to orbit insertion at Mercury, in days.
- p10-p15: Fraction of the time interval after which the DSM occurs \
in each corresponding transfer leg.
- p16-p20: Fly-by pericenter radius at each fly-by encounter, \
measured in radii of the encountered planet.
- p21-p25: B-plane rotation angles \
that orient the post-fly-by outgoing relative-velocity vector \
about the incoming relative-velocity direction at the 5 fly-by encounters, in radians.

The parameter bounds are:
- p0: [1900, 2300].
- p1: [2.5, 4.05].
- p2, p3: [0, 1] each.
- p4-p8: [100, 500] each.
- p9: [100, 600].
- p10-p15: [0.01, 0.99] each.
- p16, p17: [1.1, 6] each.
- p18-p20: [1.05, 6] each.
- p21-p25: [-\\pi, \\pi] each.\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.5 and Table 6.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L351-L357
GTOPX5_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in an impact with asteroid TW229. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Venus -> Earth -> Venus -> Earth -> Jupiter -> Saturn -> TW229 (end). \
Your objective is to maximize the change in the semi-major axis of the asteroid's orbit.

The trajectory is represented by 8 continuous parameters \
and must satisfy 6 constraints, \
which impose lower limits on the pericenter radii of the 6 fly-by maneuvers.

The 8 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Time interval from departure at Earth to the first Venus fly-by, in days.
- p2: Time interval from the first Venus fly-by to the first Earth fly-by, in days.
- p3: Time interval from the first Earth fly-by to the second Venus fly-by, in days.
- p4: Time interval from the second Venus fly-by to the second Earth fly-by, in days.
- p5: Time interval from the second Earth fly-by to the Jupiter fly-by, in days.
- p6: Time interval from the Jupiter fly-by to the Saturn fly-by, in days.
- p7: Time interval from the Saturn fly-by to impact with asteroid TW229, in days.

The parameter bounds are:
- p0: [3000, 10000].
- p1-p4: [14, 2000] each.
- p5: [100, 9000].
- p6: [366, 9000].
- p7: [300, 9000].\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.6 and Table 7.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L358-L366
GTOPX6_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in \
a rendezvous with comet 67P/Churyumov-Gerasimenko. \
The sequence of fly-by planets for this mission is as follows: \
Earth (start) -> Earth -> Mars -> Earth -> Earth -> 67P (end). \
The trajectory includes 1 deep-space maneuver (DSM) on each of the 5 transfer legs. \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 22 continuous parameters.

The 22 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Initial hyperbolic excess speed in km/s.
- p2, p3: Normalized polar-coordinate parameters \
that determine the direction of the initial hyperbolic excess velocity.
- p4: Time interval from departure at Earth to the first Earth fly-by, in days.
- p5: Time interval from the first Earth fly-by to the Mars fly-by, in days.
- p6: Time interval from the Mars fly-by to the second Earth fly-by, in days.
- p7: Time interval from the second Earth fly-by to the third Earth fly-by, in days.
- p8: Time interval from the third Earth fly-by to rendezvous with comet 67P, in days.
- p9-p13: Fraction of the time interval after which the DSM occurs \
in each corresponding transfer leg.
- p14-p17: Fly-by pericenter radius at each fly-by encounter, \
measured in radii of the encountered planet.
- p18-p21: B-plane rotation angles \
that orient the post-fly-by outgoing relative-velocity vector \
about the incoming relative-velocity direction at the 4 fly-by encounters, in radians.

The parameter bounds are:
- p0: [1460, 1825].
- p1: [3, 5].
- p2, p3: [0, 1] each.
- p4: [300, 500].
- p5, p6: [150, 800] each.
- p7: [300, 800].
- p8: [700, 1850].
- p9-p13: [0.01, 0.9] each.
- p14: [1.06, 9].
- p15-p17: [1.05, 9] each.
- p18-p21: [-\\pi, \\pi] each.\
"""


# Task and parameter semantics: Schlueter et al. (2021), Section 2.8 and Tables 9 & 10.
# https://www.sciencedirect.com/science/article/pii/S235271102100011X
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L367-L373
GTOPX7_SYSTEM_PROMPT = """\
You are an expert astrodynamicist \
specializing in interplanetary trajectory optimization. \
Your task is to design a trajectory for a multiple gravity assist space mission \
that departs from Earth and culminates in insertion into an orbit around Saturn. \
The target orbit has a pericenter radius of 108,950 km and an eccentricity of 0.98. \
As part of the trajectory design, \
you must select the 4 intermediate fly-by planets. \
Your objective is to maximize the score of the trajectory \
by minimizing the total velocity change accumulated during the entire mission.

The trajectory is represented by 10 numerical parameters \
and must satisfy 4 constraints, \
which impose lower limits on the pericenter radii of the 4 fly-by maneuvers.
The constraint thresholds remain fixed and do not depend on the selected planet codes.

The 10 parameters are ordered as follows:
- p0: Initial day, measured in days relative to 1-Jan-2000.
- p1: Time interval from departure at Earth to the first fly-by, in days.
- p2: Time interval from the first fly-by to the second fly-by, in days.
- p3: Time interval from the second fly-by to the third fly-by, in days.
- p4: Time interval from the third fly-by to the fourth fly-by, in days.
- p5: Time interval from the fourth fly-by to orbit insertion at Saturn, in days.
- p6-p9: Integer planet codes for the 4 intermediate fly-by encounters, in order.

The planet codes are:
- 1: Mercury.
- 2: Venus.
- 3: Earth.
- 4: Mars.
- 5: Jupiter.
- 6: Saturn.
- 7: Uranus.
- 8: Neptune.
- 9: Pluto.

The parameter bounds are:
- p0: [-1000, 0].
- p1: [30, 400].
- p2: [100, 470].
- p3: [30, 400].
- p4: [400, 2000].
- p5: [1000, 6000].
- p6-p9: [1, 9] each; integers only.\
"""


# Task definition: Kumar et al. (2020), Section 2.1.6.
# https://www.sciencedirect.com/science/article/pii/S2210650219308946
# Parameter semantics: Adjiman et al. (1998), Section 4.
# https://www.sciencedirect.com/science/article/pii/S0098135498000271
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L422-L428
CEC1_SYSTEM_PROMPT = """\
You are an expert chemical engineer specializing in alkylation process optimization. \
Your task is to determine the operating conditions for an alkylation unit \
that reacts an olefin feed with isobutane in the presence of an acid catalyst. \
Your objective is to maximize the daily profit generated by the alkylation unit.

The operating point is represented by 7 continuous parameters \
and must satisfy 14 constraints, \
which enforce the coupled process relationships \
required for feasible operation of the alkylation unit.

The 7 parameters are ordered as follows:
- p0: Olefin feed rate in barrels per day.
- p1: Acid addition rate in thousands of pounds per day.
- p2: Alkylate yield in barrels per day.
- p3: Acid strength in weight percent.
- p4: Motor octane number of the alkylate product.
- p5: External isobutane-to-olefin ratio.
- p6: F-4 performance number.

The parameter bounds are:
- p0: [1000, 2000].
- p1: [0, 100].
- p2: [2000, 4000].
- p3, p4: [0, 100] each.
- p5: [0, 20].
- p6: [0, 200].\
"""


# Task and parameter semantics: CEC 2020 definitions, Section 2.2.3.
# https://www.sciencedirect.com/science/article/pii/S2210650219308946
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L429-L435
CEC2_SYSTEM_PROMPT = """\
You are an expert process systems engineer \
specializing in process flowsheet optimization. \
Your task is to design a feasible process flowsheet \
using 2 continuous decision variables and 1 binary configuration decision. \
Your objective is to maximize the score of the process design \
by minimizing its nonlinear process objective.

The process design is represented by 3 numerical parameters \
and must satisfy 3 constraints, \
comprising 1 nonlinear constraint and 2 linear constraints.

The 3 parameters are ordered as follows:
- p0: First continuous process decision variable.
- p1: Second continuous process decision variable.
- p2: Binary process-configuration decision.

Parameter p2 must be integer-valued.

The parameter bounds are:
- p0: [0.2, 1].
- p1: [-2.22554, -1].
- p2: [0, 1], restricted to integers.\
"""


# Task and parameter semantics: CEC 2020 definitions, Section 2.2.1.
# https://www.sciencedirect.com/science/article/pii/S2210650219308946
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L436-L442
CEC3_SYSTEM_PROMPT = """\
You are an expert process systems engineer \
specializing in process synthesis optimization. \
Your task is to design a feasible process configuration \
using 1 continuous decision variable and 1 binary selection decision. \
Your objective is to maximize the score of the process design \
by minimizing its linear process objective.

The process design is represented by 2 numerical parameters \
and must satisfy 2 constraints, \
comprising 1 nonlinear constraint and 1 linear constraint.

The 2 parameters are ordered as follows:
- p0: Continuous process decision variable.
- p1: Binary process-selection decision.

Parameter p1 must be integer-valued.

The parameter bounds are:
- p0: [0, 1.6].
- p1: [0, 1], restricted to integers.\
"""


# Task and parameter semantics: CEC 2020 definitions, Section 2.3.6.
# https://www.sciencedirect.com/science/article/pii/S2210650219308946
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L443-L449
CEC4_SYSTEM_PROMPT = """\
You are an expert structural engineer \
specializing in truss design optimization. \
Your task is to design a symmetric 3-bar truss \
by determining the cross-sectional areas of its members. \
Your objective is to maximize the score of the truss design \
by minimizing its structural weight.

The truss is represented by 2 continuous parameters \
and must satisfy 3 nonlinear constraints, \
which impose stress limits on the 3 bars. \
The 2 symmetric diagonal bars have length sqrt(2) times the reference length, \
while the central bar has the reference length.

The 2 parameters are ordered as follows:
- p0: Cross-sectional area shared by the two symmetric diagonal bars, \
in the benchmark's area units.
- p1: Cross-sectional area of the central straight bar, \
in the benchmark's area units.

The parameter bounds are:
- p0, p1: [0, 1] each.\
"""


# Task and parameter semantics: CEC 2020 definitions, Section 2.3.5.
# https://www.sciencedirect.com/science/article/pii/S2210650219308946
# Parameter count, bounds, and constraint count:
# https://github.com/zhuyiyi-123/SOO-Bench/blob/main/soo_bench/Taskunit.py#L450-L456
CEC5_SYSTEM_PROMPT = """\
You are an expert structural engineer \
specializing in welded beam design optimization. \
Your task is to design a welded cantilever beam \
that is 14 inches long and carries an end load of 6000 pounds. \
Your objective is to maximize the score of the beam design \
by minimizing its fabrication cost.

The beam is represented by 4 continuous parameters \
and must satisfy 5 constraints, \
which impose limits on weld-to-beam geometry, normal stress, \
buckling load, weld shear stress, and tip deflection.

The 4 parameters are ordered as follows:
- p0: Weld thickness in inches.
- p1: Weld length in inches.
- p2: Beam section height in inches.
- p3: Beam section thickness in inches.

The parameter bounds are:
- p0: [0.125, 2].
- p1: [0.1, 10].
- p2: [0.1, 10].
- p3: [0.1, 2].\
"""


SOO_SYSTEM_PROMPTS = {
    "GTOPX1": GTOPX1_SYSTEM_PROMPT,
    "GTOPX2": GTOPX2_SYSTEM_PROMPT,
    "GTOPX3": GTOPX3_SYSTEM_PROMPT,
    "GTOPX4": GTOPX4_SYSTEM_PROMPT,
    "GTOPX5": GTOPX5_SYSTEM_PROMPT,
    "GTOPX6": GTOPX6_SYSTEM_PROMPT,
    "GTOPX7": GTOPX7_SYSTEM_PROMPT,
    "CEC1": CEC1_SYSTEM_PROMPT,
    "CEC2": CEC2_SYSTEM_PROMPT,
    "CEC3": CEC3_SYSTEM_PROMPT,
    "CEC4": CEC4_SYSTEM_PROMPT,
    "CEC5": CEC5_SYSTEM_PROMPT
}
