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
    # STRUCTURAL, not a fixed character window: an earlier version read 2000
    # chars after `ROUTE_ONLY = int(` and broke when an unrelated knob was added
    # between them, reporting a defect that did not exist.
    import ast
    src = open("scripts/composer/cotrain_v11.py").read()
    tree = ast.parse(src)
    blocks = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
              and n.test.id == "ROUTE_ONLY"]
    assert blocks, "no `if ROUTE_ONLY:` block at module level"
    zeroed = any(
        isinstance(st, ast.Assign)
        and any(getattr(t, "id", None) == "SUBGOAL_COST" for t in st.targets)
        and getattr(st.value, "value", None) == 0.0
        for b in blocks for st in b.body)
    assert zeroed, "a mandatory placement must not be charged"
    assert "DO NOT CHARGE FOR AN ACTION THAT IS NOT OPTIONAL" in src


def test_the_charge_still_applies_when_speaking_is_a_choice():
    """It is not deleted -- a composer that can stay silent should still be
    asked for the fewest subgoals."""
    src = open("scripts/composer/cotrain_v11.py").read()
    assert "SUBGOAL_COST = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0" in src
    assert "subgoal_cost=SUBGOAL_COST" in src, "still wired into the returns"
