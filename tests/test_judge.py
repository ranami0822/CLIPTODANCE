import pytest

from adapters import (
    BudgetExceeded,
    CachedJudge,
    Estimator,
    InvalidRequest,
    Judge,
    NoulResult,
    Question,
    QuestionKind,
    ask_in_chunks,
    choice,
    noul,
    score,
)
from adapters.cloudflare import CloudflareEstimator, CloudflareJudge
from adapters.fake import FakeJudge

IMG_A = b"image-a-bytes"
IMG_B = b"image-b-bytes"


def noul_batch(n, prefix="q"):
    return [noul(f"{prefix}{i}", f"質問{i}") for i in range(n)]


# ---------------------------------------------------------------- 質問の検証

def test_choice_needs_two_unique_options():
    with pytest.raises(InvalidRequest):
        choice("c", "向きは?", ["左"])
    with pytest.raises(InvalidRequest):
        choice("c", "向きは?", ["左", "左"])
    assert choice("c", "向きは?", ["左", "右"]).kind is QuestionKind.CHOICE


def test_score_needs_scale_and_noul_rejects_extras():
    with pytest.raises(InvalidRequest):
        score("s", "縁のきれいさ", ["1"])
    with pytest.raises(InvalidRequest):
        Question(id="n", kind=QuestionKind.NOUL, text="はい?", options=("a", "b"))


def test_duplicate_ids_in_one_batch_are_rejected():
    judge = FakeJudge()
    with pytest.raises(InvalidRequest):
        judge.ask([noul("same", "a"), noul("same", "b")])


def test_canonical_ignores_id_but_not_text():
    assert noul("a", "同じ").canonical() == noul("b", "同じ").canonical()
    assert noul("a", "同じ").canonical() != noul("a", "違う").canonical()


# ---------------------------------------------------------------- 上限とまとめ送信

def test_fake_judge_enforces_cloudflare_limits():
    judge = FakeJudge(max_questions=64, max_images=4)
    with pytest.raises(InvalidRequest):
        judge.ask(noul_batch(65))
    with pytest.raises(InvalidRequest):
        judge.ask(noul_batch(1), images=[IMG_A] * 5)


def test_ask_in_chunks_splits_at_limit_and_keeps_order():
    judge = FakeJudge(max_questions=64)
    questions = noul_batch(130)
    resp = ask_in_chunks(judge, questions)
    assert [len(c) for c in judge.calls] == [64, 64, 2]
    assert [r.question_id for r in resp.results] == [q.id for q in questions]
    assert resp.usage.api_calls == 3


def test_one_ask_is_one_api_call_not_one_per_question():
    judge = FakeJudge()
    resp = judge.ask(noul_batch(10))
    assert resp.usage.api_calls == 1 and len(judge.calls) == 1


def test_neurons_unknown_stays_none_and_known_is_summed():
    assert FakeJudge().ask(noul_batch(1)).usage.neurons is None
    judge = FakeJudge(max_questions=2, neurons_per_call=1.5)
    assert ask_in_chunks(judge, noul_batch(5)).usage.neurons == pytest.approx(4.5)


# ---------------------------------------------------------------- キャッシュ

def make_cached(tmp_path, **kw):
    inner = FakeJudge(**kw)
    return inner, CachedJudge(inner, tmp_path / "cache" / "judge.sqlite")


def test_second_identical_call_hits_cache_and_does_not_call_backend(tmp_path):
    inner, judge = make_cached(tmp_path)
    qs = noul_batch(5)
    first = judge.ask(qs, images=[IMG_A])
    second = judge.ask(qs, images=[IMG_A])
    assert len(inner.calls) == 1
    assert first.usage.api_calls == 1 and first.usage.cache_hits == 0
    assert second.usage.api_calls == 0 and second.usage.cache_hits == 5
    assert all(r.cached for r in second.results) and not any(r.cached for r in first.results)


def test_cache_ignores_question_id_and_keeps_callers_id(tmp_path):
    inner, judge = make_cached(tmp_path)
    judge.ask([noul("first", "同じ質問")], images=[IMG_A])
    resp = judge.ask([noul("second", "同じ質問")], images=[IMG_A])
    assert len(inner.calls) == 1
    assert resp.results[0].question_id == "second" and resp.results[0].cached


def test_cache_key_depends_on_image_context_and_model(tmp_path):
    inner, judge = make_cached(tmp_path)
    q = [noul("q", "同じ質問")]
    judge.ask(q, images=[IMG_A])
    judge.ask(q, images=[IMG_B])
    judge.ask(q, images=[IMG_A], context="別の状態")
    assert len(inner.calls) == 3
    other_inner = FakeJudge(model_id="other")
    other = CachedJudge(other_inner, tmp_path / "cache" / "judge.sqlite")
    other.ask(q, images=[IMG_A])
    assert len(other_inner.calls) == 1  # 同じDBでも model_id が違えば別のキー


def test_partial_hit_only_asks_for_misses_and_preserves_order(tmp_path):
    inner, judge = make_cached(tmp_path)
    judge.ask([noul("a", "既出1"), noul("b", "既出2")], images=[IMG_A])
    qs = [noul("n1", "新規1"), noul("a", "既出1"), noul("n2", "新規2"), noul("b", "既出2")]
    resp = judge.ask(qs, images=[IMG_A])
    assert [q.text for q in inner.calls[-1]] == ["新規1", "新規2"]
    assert [r.question_id for r in resp.results] == ["n1", "a", "n2", "b"]
    assert [r.cached for r in resp.results] == [False, True, False, True]


def test_cached_judge_chunks_large_batches_itself(tmp_path):
    inner, judge = make_cached(tmp_path, max_questions=64)
    resp = judge.ask(noul_batch(130))
    assert [len(c) for c in inner.calls] == [64, 64, 2]
    assert len(resp.results) == 130


def test_cached_results_roundtrip_all_kinds(tmp_path):
    inner, judge = make_cached(tmp_path)
    qs = [noul("n", "はい?"), choice("c", "向き", ["左", "右", "不明"]), score("s", "縁", ["1", "2", "3"])]
    first = judge.ask(qs)
    second = judge.ask(qs)
    for a, b in zip(first.results, second.results):
        assert type(a) is type(b)
    assert second.results[1].probs == first.results[1].probs
    assert second.results[2].level_probs == first.results[2].level_probs


def test_failures_propagate_and_are_not_cached(tmp_path):
    inner, judge = make_cached(tmp_path, fail_with=BudgetExceeded("無料枠を超えた"))
    with pytest.raises(BudgetExceeded):
        judge.ask([noul("q", "質問")])
    inner.fail_with = None  # 復旧
    resp = judge.ask([noul("q", "質問")])
    assert isinstance(resp.results[0], NoulResult) and not resp.results[0].cached


# ---------------------------------------------------------------- Cloudflare の骨組み

def test_cloudflare_adapters_follow_protocols_and_documented_limits():
    judge, est = CloudflareJudge(), CloudflareEstimator()
    assert isinstance(judge, Judge) and isinstance(est, Estimator)
    assert judge.capabilities.max_questions == 64 and judge.capabilities.max_images == 4
    with pytest.raises(NotImplementedError):
        judge.ask(noul_batch(1))
