"""Finite launch ownership must not leak or retire resumed conversations."""
import argparse
from contextlib import asynccontextmanager

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code,resume,source,expected', [
    (0, False, 'oneshot', True),
    (3, False, 'oneshot', False),
    (2, False, 'oneshot', False),
    (130, False, 'oneshot', False),
    (0, True, 'oneshot', False),
    (0, False, 'cli', False),
])
async def test_finite_launch_retires_only_its_successful_ephemeral_session(
    monkeypatch, tmp_path, exit_code, resume, source, expected
):
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []
    snapshot = {'stored_session_id': 'physical', 'subscription_id': 'viewer'}

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == 'runtime.describe':
                return {'session_create': {'sources': ['oneshot', 'cli'], 'parameters': ['cwd']}}
            return snapshot

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        return exit_code

    monkeypatch.setattr(gateway_chat, 'connect_gateway', connected)
    monkeypatch.setattr(GatewayChatView, 'run', rendered)
    monkeypatch.setattr(gateway_chat, 'check_resume_policy', lambda *a: None)
    monkeypatch.setattr(gateway_chat, 'env_launch_flags', dict)
    monkeypatch.setattr(gateway_chat, 'client_env_policy', lambda *a: {})
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(oneshot='readiness review', source=source,
                              resume='physical' if resume else None)
    assert await gateway_chat.run_gateway_chat(args) == exit_code
    closes = [params for method, params in calls if method == 'session.detach']
    assert closes == ([{'session_id': 'physical', 'subscription_id': 'viewer',
                        'end_reason': 'cli_exit'}] if expected else [])
