import os
from unittest.mock import patch

import pytest

from codeboarding_cli.commands.chatgpt_auth import run_from_args
from codeboarding_cli.main import _inject_default_subcommand, build_parser, main
from agents.llm_config import initialize_agent_llm, validate_api_key_provided
from agents.chatgpt_plan import ChatGPTPlanChatModel
from infra.chatgpt_auth import AUTH_ENV, ChatGPTAuthError


def args(*arguments):
    return build_parser().parse_args(["auth", "chatgpt", *arguments])


def test_auth_subcommand_is_not_a_repository_url():
    assert _inject_default_subcommand(["auth", "chatgpt", "status"]) == ["auth", "chatgpt", "status"]


def test_status_never_starts_analysis_or_prints_tokens(tmp_path, capsys):
    with (
        patch.dict(os.environ, {AUTH_ENV: str(tmp_path / "auth.json")}, clear=True),
        patch("codeboarding_cli.main.full_analysis.run_from_args", side_effect=AssertionError("analysis started")),
    ):
        main(["auth", "chatgpt", "status"])
    assert "No saved ChatGPT accounts" in capsys.readouterr().out


def test_login_passes_options_and_sets_auth_file(tmp_path, capsys):
    path = tmp_path / "accounts.json"
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "codeboarding_cli.commands.chatgpt_auth.ChatGPTAccounts.login",
            return_value={"email": "test@example.com", "client_id": "oaiapp_test"},
        ) as login,
    ):
        run_from_args(
            args("login", "--auth-file", str(path), "--no-browser", "--new-account", "--port", "1455"), build_parser()
        )
        assert os.environ[AUTH_ENV] == str(path)
        login.assert_called_once_with(new_account=True, account="", port=1455, open_browser=False)
    assert "ChatGPT plan usage enabled" in capsys.readouterr().out


def test_invalid_port_is_rejected(tmp_path):
    with patch.dict(os.environ, {AUTH_ENV: str(tmp_path / "auth.json")}):
        with pytest.raises(SystemExit) as error:
            run_from_args(args("login", "--port", "65536"), build_parser())
    assert error.value.code == 2


def test_missing_login_is_actionable(tmp_path, capsys):
    with patch.dict(os.environ, {AUTH_ENV: str(tmp_path / "auth.json")}):
        with pytest.raises(SystemExit) as error:
            run_from_args(args("models"), build_parser())
    assert error.value.code == 1
    assert "login" in capsys.readouterr().err


def test_plan_requires_credentials_in_normal_provider_validation(tmp_path):
    with patch.dict(os.environ, {AUTH_ENV: str(tmp_path / "auth.json")}, clear=True):
        with pytest.raises(ChatGPTAuthError, match="login"):
            validate_api_key_provided()


def test_analysis_builds_dedicated_provider_without_api_key(tmp_path):
    with patch.dict(os.environ, {AUTH_ENV: str(tmp_path / "auth.json")}, clear=True):
        model = initialize_agent_llm("account-model")
        assert isinstance(model, ChatGPTPlanChatModel)
        assert model.model_name == "account-model"
        assert model.auth_file == str(tmp_path / "auth.json")
        assert "temperature" not in model.model_dump()
