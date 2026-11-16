"""The benchmark itself: gold SQL runs, and typical mistakes are scored as wrong."""
import pytest

from evaluation.bench import attach_gold, load_questions
from evaluation.compare import results_match, set_country_names


@pytest.fixture(scope="module")
def questions(db):
    set_country_names(db.run("select geo, country_name from dim_country").rows)
    return {q.id: q for q in attach_gold(load_questions(), db)}


def test_counts(questions):
    test = [q for q in questions.values() if q.split == "test"]
    assert len(test) >= 60
    assert {q.difficulty for q in test} == {"easy", "medium", "hard"}


def test_gold_is_not_empty(questions):
    for q in questions.values():
        assert q.gold.rows, q.id


def _score(db, q, sql):
    r = db.run(sql)
    assert r.ok, r.error
    return results_match(r.columns, r.rows, q.gold.columns, q.gold.rows, q.order_matters)


def test_known_mistakes_are_wrong(db, questions):
    # Forgetting the EU27 aggregate row.
    assert not _score(db, questions["t21"],
                      "select geo from metrics_country_year where year = 2024 order by new_electric desc nulls last limit 1")
    # Summing the fuel hierarchy across levels (double counting).
    assert not _score(db, questions["t31"],
                      "select sum(units) from fct_new_lorries_bi "
                      "where geo='DE' and year=2024 and mot_nrg in ('PET', 'PET_X_HYB', 'ELC_PET_HYB')")
    # Answering a missing EU27 value with the partial sum.
    assert not _score(db, questions["t36"],
                      "select partial_new_electric::double / partial_new_total from metrics_eu27_year where year = 2025")


def test_equivalent_answers_are_right(db, questions):
    # Country names instead of codes, percentage instead of fraction, extra columns.
    assert _score(db, questions["t22"],
                  "select c.country_name, m.new_electric, round(100*m.new_electric_share, 1) as pct "
                  "from metrics_country_year m join dim_country c using (geo) "
                  "where c.is_eu27 and m.year = 2024 order by m.new_electric desc nulls last limit 5")
    assert _score(db, questions["t03"],
                  "select round(100.0 * new_electric / new_total, 2) from metrics_country_year where geo='DK' and year=2024")
