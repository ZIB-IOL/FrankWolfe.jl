@enum LSMethod agnostic = 1 backtracking = 2 goldenratio = 3 nonconvex = 4 shortstep = 5 fixed = 6 rationalshortstep =
    7 adaptive = 8
@enum Emph blas = 1 memory = 2
@enum StepType initial = 1 regular = 2 lazy = 3 lazylazy = 4 dualstep = 5 away = 6 last = 1000 pp = 1001

const st = (initial="I", regular="FW", lazy="L", lazylazy="LL", dualstep="LD", away="A", last="Last", pp="PP")
