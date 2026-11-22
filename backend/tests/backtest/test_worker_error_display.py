from types import SimpleNamespace

import pytest

from app.backtest import worker


def test_worker_traceback_is_logged_but_not_returned_to_user(monkeypatch,caplog):
    class Events:
        def get(self,timeout):
            return {'type':'error','message':'请将买卖口径设为次日开盘','traceback':'Traceback: /app/private/source.py'}
        def close(self):
            pass
        def join_thread(self):
            pass
    process=SimpleNamespace(start=lambda:None,is_alive=lambda:False,join=lambda timeout:None,exitcode=1,pid=None)
    context=SimpleNamespace(Queue=Events,Event=lambda:None,Process=lambda **kwargs:process)
    monkeypatch.setattr(worker.mp,'get_context',lambda _method:context)
    monkeypatch.setattr(worker,'_effective_rss_limit_bytes',lambda:0)
    with pytest.raises(worker.BacktestWorkerError) as caught:
        worker.run_worker_task({'kind':'backtest'})
    assert str(caught.value)=='请将买卖口径设为次日开盘'
    assert 'Traceback' in caplog.text and '/app/private/source.py' in caplog.text
