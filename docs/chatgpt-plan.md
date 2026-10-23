# Use your ChatGPT plan in CodeBoarding

This experimental provider uses OpenAI's official **Sign in with ChatGPT** flow
and the public Responses API. Eligible accounts can use their ChatGPT plan
allowance or available ChatGPT credits without an OpenAI API key. Access and
limits depend on the selected account/workspace. No request falls back to API-key
billing when login, eligibility, or quota checks fail.

## Try this branch on your PC

Use Python **3.12** and [uv](https://docs.astral.sh/uv/). From an existing clone of
`jvnsg/CodeBoarding`:

```sh
git fetch origin
git switch --track origin/feat/chatgpt-plan-provider
uv sync --frozen
uv run codeboarding auth chatgpt login
uv run codeboarding auth chatgpt models
uv run codeboarding auth chatgpt test
uv run codeboarding full --chatgpt-plan --local /path/to/your/repository --render html
```

If you already have the branch locally, use `git switch feat/chatgpt-plan-provider`
and `git pull --ff-only` instead of `git switch --track`.
For a new clone, run:

```sh
git clone --branch feat/chatgpt-plan-provider https://github.com/jvnsg/CodeBoarding.git
cd CodeBoarding
uv sync --frozen
```

On Windows, the same `uv run` commands work in PowerShell without activating a
virtual environment. Replace the repository argument with e.g.
`--local 'C:\Users\Jens\Projects\my-repo'`.
Analysis bootstraps the repository's usual language-server tools on first use;
login and the connection test do not install language servers.

The login opens your system browser and listens **only on 127.0.0.1** on an
available port. Sign in, choose the workspace, and explicitly allow ChatGPT plan
usage. Keep the terminal running until it confirms the connection.
`--no-browser` prints the authorization URL if automatic opening is unavailable;
open that URL on the **same computer** as the CLI. A fixed port is available with
`login --port 1455`.

The default model is `gpt-6.1-sol`. List account-specific models with `models`, then
use `test --model <slug>` to check another one. For analysis, override the model
in `~/.codeboarding/config.toml`:

```toml
[llm]
agent_model = "<slug from the models command>"
```

Or set the existing `AGENT_MODEL` environment variable.
If another provider key/URL is already configured in the shell or config file,
remove that provider selection before running analysis with `--chatgpt-plan`.
CodeBoarding deliberately rejects ambiguous provider configuration.

## Keep this provider selected

Instead of passing `--chatgpt-plan` on each full, incremental, or partial run:

```toml
[provider]
chatgpt_plan_auth_file = "~/.codeboarding/chatgpt/accounts.json"
```

Only the file **path** belongs in config.toml, never the tokens. You can also set
`CHATGPT_PLAN_AUTH_FILE`. For an alternate credential file, pass
`auth chatgpt login --auth-file /path/to/accounts.json` and select that same path
for analysis using config.toml or the environment variable.

## Accounts and sessions

```sh
uv run codeboarding auth chatgpt status
uv run codeboarding auth chatgpt login --new-account
uv run codeboarding auth chatgpt login --account oaiapp_YOUR_SAVED_CLIENT_ID
uv run codeboarding auth chatgpt logout
```

Each account/workspace registration is stored separately. `status` shows account
labels and issued client IDs, never tokens. Login defaults to the active
registration; `--new-account` starts a separate registration. Logout revokes the
active renewable session and clears its tokens while retaining its registration
for later sign-in. Access can also be managed in ChatGPT **Settings > Usage**.

Tokens are kept in `~/.codeboarding/chatgpt/accounts.json`, outside the repo.
Writes are atomic; Unix files are owner-readable/writable only (0600), with a
0700 directory on creation. On Windows, keep this under your private user
profile and protect it with your normal account ACLs. Tokens refresh
automatically, with a cross-process file lock to serialize refresh-token
rotation. Never commit, paste, or attach this credential file to an issue.

For a personally operated VM, follow OpenAI's
[self-hosted VM procedure](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
A loopback callback in your PC's browser cannot reach a callback listener on a
remote server. Initial testing is simplest directly on your PC.

## Implementation and current limits

- Existing analysis agents and output formats are retained. A dedicated
  LangChain adapter translates CodeBoarding's messages, repository tools, and
  structured-result tools to Responses requests with a `codeboarding` namespace.
- Requests always use `store: false` and `stream: true`. The adapter consumes SSE
  through completion, returning an ordinary LangChain result. It retains
  encrypted reasoning and tool-call IDs for subsequent turns. Token usage is
  reported to the existing callbacks. This is not incremental UI token streaming.
- Unsupported preview options, such as temperature or output-token caps, are
  not sent. Hosted tools, transcription, embeddings, and image generation are
  outside this provider's scope.
- Quota exhaustion stops analysis; it never silently changes billing provider.
- OAuth, SDK streaming, tool execution and LangChain structured results have
  automated offline tests. A real account login and live analysis must still be
  verified by the user; automated tests cannot establish account eligibility.

Official references:
[registration](https://developers.openai.com/siwc/token-sharing-open-source/sign-in),
[models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference),
[preview limits](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).
