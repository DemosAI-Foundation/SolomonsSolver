"""Tests for context isolation (Solomon's Paradox implementation)."""
from src.llm.context_manager import ContextManager


class TestContextManager:
    def test_fresh_windows_are_separate(self):
        cm = ContextManager()
        w1 = cm.create_fresh_context("t1", purpose="critique")
        w2 = cm.create_fresh_context("t1", purpose="cove_answer")
        assert w1.window_id != w2.window_id
        assert w1.messages == [] and w2.messages == []
        assert w1.isolated and w2.isolated

    def test_inject_task_only_user_message(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="critique")
        cm.inject_task_only(w, task="TASK", trace="TRACE", answer="ANS")
        assert len(w.messages) == 1
        assert w.messages[0]["role"] == "user"
        content = w.messages[0]["content"]
        assert "TASK" in content and "TRACE" in content and "ANS" in content

    def test_inject_without_trace_or_answer(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="cove_answer")
        cm.inject_task_only(w, task="only task")
        content = w.messages[0]["content"]
        assert "only task" in content
        assert "REASONING TRACE" not in content
        assert "ANSWER:" not in content

    def test_windows_for_task_scoped(self):
        cm = ContextManager()
        cm.create_fresh_context("t1", purpose="a")
        cm.create_fresh_context("t2", purpose="b")
        cm.create_fresh_context("t1", purpose="c")
        assert len(cm.windows_for_task("t1")) == 2
        assert len(cm.windows_for_task("t2")) == 1

    def test_verify_isolation_clean(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="critique")
        cm.inject_task_only(w, task="T", trace="TR", answer="A")
        report = cm.verify_isolation("t1")
        assert report["isolated"] is True
        assert report["violations"] == []
        assert report["window_count"] == 1

    def test_verify_isolation_detects_assistant_history(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="critique")
        w.messages.append({"role": "assistant", "content": "leaked history"})
        report = cm.verify_isolation("t1")
        assert report["isolated"] is False
        assert any("assistant" in v for v in report["violations"])

    def test_verify_isolation_detects_non_isolated_flag(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="critique")
        w.isolated = False
        report = cm.verify_isolation("t1")
        assert report["isolated"] is False

    def test_clear_by_task(self):
        cm = ContextManager()
        cm.create_fresh_context("t1", purpose="a")
        cm.create_fresh_context("t2", purpose="b")
        cm.clear("t1")
        assert cm.windows_for_task("t1") == []
        assert len(cm.windows_for_task("t2")) == 1

    def test_clear_all(self):
        cm = ContextManager()
        cm.create_fresh_context("t1", purpose="a")
        cm.clear()
        assert len(cm.windows_for_task("t1")) == 0

    def test_render(self):
        cm = ContextManager()
        w = cm.create_fresh_context("t1", purpose="x")
        cm.inject_task_only(w, task="T")
        assert "[user]" in w.render()
