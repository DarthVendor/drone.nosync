"""A router cannot decline to place, so it must not be charged per placement.

`SUBGOAL_COST` exists so a composer that CHOOSES when to speak reaches the goal
with the fewest subgoals. Under `route_only` EOS is masked and it places at
every report (~37-63 a flight), and `returns_from_stream` folds that charge into
the DISCOUNTED RETURN at every future decision: at gamma 0.99 over ~90 reports
about 177 against a cost scale of ~900. The only way to reduce it is to end the
flight, and crashing is far easier than arriving.

MEASURED once the estimator was otherwise sound (EV +0.292, kl inside the cap,
policy stepping): arrive 0.383 -> 0.273, crash 0.610 -> 0.725, and speak
0.401 -> 0.187 -- the gradient pushing toward a silence that is masked and
cannot be taken. The objective was paying the composer to stop flying.
"""


def test_route_only_zeroes_the_per_placement_charge():
    src = open("scripts/composer/cotrain_v11.py").read()
    i = src.index("ROUTE_ONLY = int(")
    tail = src[i:i + 2000]
    assert "SUBGOAL_COST = 0.0" in tail, \
        "a mandatory placement must not be charged"
    assert "DO NOT CHARGE FOR AN ACTION THAT IS NOT OPTIONAL" in tail


def test_the_charge_still_applies_when_speaking_is_a_choice():
    """It is not deleted -- a composer that can stay silent should still be
    asked for the fewest subgoals."""
    src = open("scripts/composer/cotrain_v11.py").read()
    assert "SUBGOAL_COST = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0" in src
    assert "subgoal_cost=SUBGOAL_COST" in src, "still wired into the returns"
