import re

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.model_ids import ModelIdentifier
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import ChatContext
from mellea.stdlib.chat import Message
from mellea.stdlib.requirement import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def setup_local_session(
    *,
    model_id: ModelIdentifier = model_ids.OPENAI_GPT_OSS_20B,
    system_prompt: str = "You are a helpful assistant.",
) -> MelleaSession:
    m = MelleaSession(
        # ctx=mellea.stdlib.base.LinearContext(),
        ctx=ChatContext(),
        backend=OllamaModelBackend(model_id=model_id),
    )

    # Add the system prompt and the goal to the chat history.
    # m.ctx.insert(mellea.stdlib.chat.Message(role="system", content=system_prompt))
    m.ctx.add(Message(role="system", content=system_prompt))

    return m


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
        for i, _ in enumerate(m.ctx.linearize()):
            print(i, ": ", _)
    except Exception:
        print("fail ...")


if __name__ == "__main__":
    main()
