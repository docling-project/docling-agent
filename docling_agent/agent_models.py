import re

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from tabulate import tabulate  # type: ignore[import-untyped]

from docling_agent.log import logger

# Use shared logger from docling_agent.agents

# Whether to log LLM requests/responses — toggled by the orchestrator via configure_llm_logging().
_log_llm_io: bool = False


def configure_llm_logging(enabled: bool) -> None:
    """Enable or disable per-call LLM request/response logging."""
    global _log_llm_io
    _log_llm_io = enabled


class _LoggingSession:
    """Thin wrapper around MelleaSession that logs every instruct call."""

    def __init__(self, session: MelleaSession) -> None:
        self._s = session

    # Forward attribute access to the wrapped session
    def __getattr__(self, name: str):
        return getattr(self._s, name)

    def instruct(self, prompt: str, **kwargs):
        if _log_llm_io:
            logger.debug(f"[LLM REQUEST]\n{prompt}")
        result = self._s.instruct(prompt, **kwargs)
        if _log_llm_io:
            logger.debug(f"[LLM RESPONSE]\n{result.value}")
        return result


def setup_local_session(
    *,
    model_id: ModelIdentifier = model_ids.OPENAI_GPT_OSS_20B,
    system_prompt: str = "You are a helpful assistant.",
) -> _LoggingSession:
    ctx = ChatContext()
    ctx = ctx.add(Message(role="system", content=system_prompt))

    m = MelleaSession(
        ctx=ctx,
        backend=OllamaModelBackend(model_id=model_id),
    )

    return _LoggingSession(m)


def view_linear_context(m: "MelleaSession | _LoggingSession"):
    rows = []
    for i, _ in enumerate(m.ctx.view_for_generation()):  # type: ignore[arg-type]
        if isinstance(_, Message):
            if len(_.content) > 64:
                rows.append([i, _.role, (_.content[0:32] + " ... " + _.content[-32:])])
            else:
                rows.append([i, _.role, _.content])
        else:
            rows.append([i, "<unknown>", str(_)[0:64]])

    logger.info(
        f"linearized chat:\n\n {tabulate(rows, headers=['turn', 'role', 'message'])}"
    )


def matches_html_code_block(text: str) -> bool:
    """
    Check if a string matches the pattern ```html(.*)?```

    Args:
        text (str): The string to check

    Returns:
        bool: True if the string matches the pattern, False otherwise
    """
    pattern = r"^```html.*```$"
    print("input: ", text)
    result = bool(re.match(pattern, text, re.DOTALL))
    print(result)

    return result


def main():
    """
    m = MelleaSession(
        ctx=mellea.stdlib.base.LinearContext(),
        backend=OllamaModelBackend(
            model_id=model_ids.OPENAI_GPT_OSS_20B #, model_options={ModelOption.SEED: 42}
        )
    )

    # Add the system prompt and the goal to the chat history.
    m.ctx.insert(mellea.stdlib.chat.Message(role="system", content="You are an expert material-scientist."))
    """

    m = setup_local_session(system_prompt="You are an expert material-scientist.")

    _ = m.instruct(
        "Please write me a table in HTML with the most common properties of the most common polymers. The polymers need to be rows, the properties need to be colums.",
        requirements=[
            # "The resulting HTML table should satisfy the following regex ```html(.*)?```",
            Requirement(
                description="The resulting HTML table should satisfy the following regex ```html(.*)?```",
                validation_fn=simple_validate(matches_html_code_block),
            )
        ],
        # user_variables={"name": name, "notes": notes},
        strategy=RejectionSamplingStrategy(loop_budget=5),
    )
    # print(answer)

    try:
        for i, _ in enumerate(m.ctx.view_for_generation()):  # type: ignore[arg-type]
            print(i, ": ", _)
    except Exception:
        print("fail ...")


if __name__ == "__main__":
    main()
