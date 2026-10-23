"""Manage ChatGPT plan login without bootstrapping analysis or language servers."""

import argparse
import os
from pathlib import Path

import requests

from agents.chatgpt_plan import ChatGPTPlanChatModel, ChatGPTPlanError
from openai import OpenAIError
from infra.chatgpt_auth import AUTH_ENV, DEFAULT_AUTH_PATH, RESOURCE, ChatGPTAccounts, ChatGPTAuthError


def add_arguments(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("auth", help="Manage model-provider authentication")
    providers = parser.add_subparsers(dest="auth_provider", required=True)
    chatgpt = providers.add_parser("chatgpt", help="Use your ChatGPT plan (OAuth)")
    chatgpt.add_argument("action", choices=["login", "status", "models", "test", "logout"])
    chatgpt.add_argument("--auth-file", type=Path, help=f"Credential file (default: {DEFAULT_AUTH_PATH})")
    chatgpt.add_argument("--new-account", action="store_true", help="Register another ChatGPT account/workspace")
    chatgpt.add_argument("--account", default="", help="Saved client ID to sign in again")
    chatgpt.add_argument("--port", type=int, default=0, help="Loopback callback port (default: automatic)")
    chatgpt.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    chatgpt.add_argument("--model", default="gpt-6.1-sol", help="Model for the connection test")


def run_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.auth_file:
        os.environ[AUTH_ENV] = str(args.auth_file.expanduser())
    accounts = ChatGPTAccounts()
    try:
        if args.action == "login":
            if not 0 <= args.port <= 65535:
                parser.error("--port must be between 0 and 65535")
            profile = accounts.login(
                new_account=args.new_account, account=args.account, port=args.port, open_browser=not args.no_browser
            )
            print(f"Connected: {profile.get('email') or profile['subject']} ({profile['client_id']})")
            print("ChatGPT plan usage enabled. Run analysis with --chatgpt-plan.")
        elif args.action == "status":
            data = accounts.read()
            if not data["profiles"]:
                print("No saved ChatGPT accounts. Run: codeboarding auth chatgpt login")
            for client_id, profile in data["profiles"].items():
                marker = "active" if client_id == data["active"] else "saved"
                connection = "connected" if profile.get("refresh_token") else "signed out"
                print(f"{marker}: {profile.get('email') or profile.get('subject', '')} ({client_id}), {connection}")
        elif args.action == "models":
            response = requests.get(
                RESOURCE + "/models", headers={"Authorization": "Bearer " + accounts.access_token()}, timeout=30
            )
            if response.status_code != 200:
                raise ChatGPTAuthError(f"Model catalog unavailable (HTTP {response.status_code}).")
            for model in response.json()["models"]:
                if model.get("visibility") == "list":
                    print(f"{model['slug']}\t{model.get('display_name', model['slug'])}")
        elif args.action == "test":
            answer = ChatGPTPlanChatModel(model_name=args.model).invoke("Say exactly: CodeBoarding connected.")
            print(answer.content)
            print("Completed using your ChatGPT plan. No API-key fallback was used.")
        elif args.action == "logout":
            accounts.logout()
            print("ChatGPT session signed out.")
    except (ChatGPTAuthError, ChatGPTPlanError, OpenAIError, requests.RequestException, OSError) as exc:
        parser.exit(1, f"CodeBoarding: {exc}\n")
