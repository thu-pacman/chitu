from typing import List, Optional

import fire

from cinfer import Dialog, Llama

from torch.profiler import profile, record_function, ProfilerActivity

from cinfer.global_vars import set_global_variables, get_timers

import timeit

set_global_variables()


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
        Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
        1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
        2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
        3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
        These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        # [
        #     {"role": "system", "content": "Always answer with Haiku"},
        #     {"role": "user", "content": "I am going to Paris, what should I see?"},
        # ],
        # [
        #     {
        #         "role": "system",
        #         "content": "Always answer with emojis",
        #     },
        #     {"role": "user", "content": "How to go from Beijing to NY?"},
        # ],
    ]

    t = timeit.timeit(lambda: generator.prefill(dialogs), number=3) / 3
    print(t)

    t = timeit.timeit(lambda: generator.prefill(dialogs), number=3) / 3
    print(t)

    t = timeit.timeit(lambda: generator.prefill(dialogs), number=3) / 3
    print(t)

    # def add_to_logging(name):
    #     if name in timers.timers:
    #         timers_to_log.append(name)

    # for i in range(3):
    #     results = generator.chat_completion(
    #         dialogs,
    #         max_gen_len=max_gen_len,
    #         temperature=temperature,
    #         top_p=top_p,
    #     )
    #     timers = get_timers()
    #     timers_to_log = []

    #     add_to_logging("attn")
    #     add_to_logging("feed")
    #     print(timers_to_log)
    #     timers.log(timers_to_log)

    #     for dialog, result in zip(dialogs, results):
    #         for msg in dialog:
    #             print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #         print(
    #             f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #         )
    #         print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
