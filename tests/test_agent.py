"""Agent loop behaviour with a scripted LLM (no API calls)."""
import itertools

from agent.agent import Agent, AgentConfig, SingleShot
from agent.llm import ScriptedClient, ToolCall, Turn

_ids = itertools.count()


def call(name, **args):
    return ToolCall(id=f"c{next(_ids)}", name=name, arguments=args)


def script(*turns):
    it = iter(turns)
    return ScriptedClient(lambda events: next(it))


def test_happy_path(db):
    llm = script(
        Turn("", [call("run_sql", sql="select new_electric from metrics_country_year where geo='DE' and year=2024")]),
        Turn("", [call("final_answer", query_id="q1", answer="Germany registered 2,978 electric lorries in 2024.")]),
    )
    res = Agent(llm, db).run("How many in Germany in 2024?")
    assert res.status == "answered" and res.final_query.rows == [(2978,)]
    assert res.grounding.ok and res.n_llm_calls == 2


def test_retry_after_sql_error(db):
    llm = script(
        Turn("", [call("run_sql", sql="select electric from metrics_country_year")]),
        Turn("", [call("run_sql", sql="select new_electric from metrics_country_year where geo='DE' and year=2024")]),
        Turn("", [call("final_answer", query_id="q2", answer="2978")]),
    )
    res = Agent(llm, db).run("q")
    assert res.status == "answered" and res.n_sql_errors == 1 and res.final_query.ok


def test_ungrounded_number_is_sent_back(db):
    llm = script(
        Turn("", [call("run_sql", sql="select new_electric from metrics_country_year where geo='DE' and year=2024")]),
        Turn("", [call("final_answer", query_id="q1", answer="Germany registered 3,100 electric lorries.")]),
        Turn("", [call("final_answer", query_id="q1", answer="Germany registered 2,978 electric lorries.")]),
    )
    res = Agent(llm, db).run("q")
    assert res.grounding_rejections == 1 and res.grounding.ok and "2,978" in res.answer


def test_retry_budget(db):
    bad = [Turn("", [call("run_sql", sql="select nope from dim_year")]) for _ in range(4)]
    llm = script(*bad, Turn("I could not answer.", []), Turn("Still no.", []))
    res = Agent(llm, db, AgentConfig(max_sql_failures=3)).run("q")
    assert res.n_sql_errors == 3          # the 4th call is refused before running
    assert res.status == "no_final_answer" and res.final_query is None


def test_write_attempt_is_blocked(db):
    llm = script(
        Turn("", [call("run_sql", sql="drop table dim_year")]),
        Turn("", [call("run_sql", sql="select count(*) from dim_year")]),
        Turn("", [call("final_answer", query_id="q2", answer="13 years.")]),
    )
    res = Agent(llm, db).run("q")
    assert "Only SELECT" in res.queries[0].error and res.final_query.rows == [(13,)]


def test_single_shot(db):
    llm = script(Turn("```sql\nselect count(*) from dim_year\n```"))
    res = SingleShot(llm, db).run("How many years?")
    assert res.final_query.rows == [(13,)]


def test_chart(db, tmp_path):
    llm = script(
        Turn("", [call("run_sql", sql="select year, new_electric from metrics_eu27_year where year <= 2024")]),
        Turn("", [call("make_chart", query_id="q1", kind="line", x="year", y="new_electric", title="EU27")]),
        Turn("", [call("final_answer", query_id="q1", answer="See chart.")]),
    )
    res = Agent(llm, db, AgentConfig(chart_dir=tmp_path)).run("trend?")
    assert len(res.charts) == 1 and (tmp_path / "chart_1.png").exists()


def test_final_answer_in_same_turn_as_query_is_refused(db):
    llm = script(
        Turn("", [call("run_sql", sql="select count(*) from dim_year"),
                  call("final_answer", query_id="q1", answer="There are 99 years.")]),
        Turn("", [call("final_answer", query_id="q1", answer="There are 13 years.")]),
    )
    res = Agent(llm, db).run("q")
    assert res.answer == "There are 13 years." and res.n_llm_calls == 2
