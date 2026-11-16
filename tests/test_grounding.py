from agent.grounding import check_grounding, extract_numbers


def test_extract():
    nums = extract_numbers("Germany had 2,978 of 5,575 (53.4%) in EU27_2020; see q3.")
    assert [n.value for n in nums] == [2978, 5575, 53.4]
    assert nums[2].is_percent


def test_grounded_with_rounding_and_percent():
    rep = check_grounding("Denmark reached 17.3% in 2024, with 404 lorries.", [0.17331, 404, 2024])
    assert rep.ok, rep.to_dict()


def test_invented_number_is_flagged():
    rep = check_grounding("The share rose by 4.6 percentage points to 17.3%.", [0.17331, 0.1271])
    assert [m.text for m in rep.ungrounded] == ["4.6"]


def test_question_numbers_and_small_counts_allowed():
    rep = check_grounding("4 countries had at least 5% in 2024.", [4], question="at least 5% in 2024")
    assert rep.ok
