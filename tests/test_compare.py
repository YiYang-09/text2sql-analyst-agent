from evaluation.compare import results_match, set_country_names, values_match

set_country_names([("DE", "Germany"), ("DK", "Denmark"), ("EL", "Greece")])


def test_scalar_exact_and_rounding():
    assert values_match(5575, 5575)
    assert not values_match(5576, 5575)
    assert values_match(0.04120300, 0.041203)
    assert values_match(4.12, 4.12030)          # percent rounded to 2 decimals (via scale in results_match)
    assert not values_match(4.0, 4.1203)        # rounding to 0 decimals loses too much


def test_percent_vs_fraction():
    assert results_match(["pct"], [(4.12,)], ["share"], [(0.041203,)])
    assert results_match(["share"], [(0.0412,)], ["share"], [(0.041203,)])
    assert not results_match(["share"], [(0.05,)], ["share"], [(0.041203,)])


def test_column_order_names_and_extra_columns():
    gold = [("DE", 2978), ("DK", 600)]
    pred = [(600, "Denmark", "x"), (2978, "Germany", "y")]
    assert results_match(["n", "country", "extra"], pred, ["geo", "new_electric"], gold)


def test_row_count_and_order():
    gold = [("DE",), ("DK",)]
    assert not results_match(["geo"], [("DE",)], ["geo"], gold)
    assert results_match(["geo"], [("DK",), ("DE",)], ["geo"], gold)
    assert not results_match(["geo"], [("DK",), ("DE",)], ["geo"], gold, order_matters=True)


def test_nulls_and_row_alignment():
    gold = [("DE", 1), ("DK", None)]
    assert results_match(["g", "v"], [("DK", None), ("DE", 1)], ["g", "v"], gold)
    assert not results_match(["g", "v"], [("DK", 1), ("DE", None)], ["g", "v"], gold)
    assert values_match("greece", "EL")
