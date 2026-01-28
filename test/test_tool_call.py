import subprocess, sys, asyncio, os, json, signal, logging, itertools, time
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)

logging.basicConfig(format="[%(name)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("tester")
logger_engine = logging.getLogger("engine")
sys.stdout.reconfigure(line_buffering=True)


MAX_CONCURRENT = 32
MAX_RETRY = 10
CHOICE_FUNC_NAME = "get_temperature"
CHOICE_FUNC = {"type": "function", "function": {"name": CHOICE_FUNC_NAME}}
LAUNCH_TIMEOUT = 300
REQ_TIMEOUT = 300

# nums: describe number of tools called in each round. list of set of int, list represents chat rounds, set represents called tools
CASES = [
    dict(prompt=0, choice="none", parallel=False, nums=[{0}]),
    dict(prompt=0, choice="none", parallel=True, nums=[{0}]),
    dict(prompt=0, choice="required", parallel=False, nums=[{1}]),
    dict(prompt=0, choice="required", parallel=True, nums=[{1, 2}]),
    dict(prompt=0, choice="auto", parallel=False, nums=[{0}]),
    dict(prompt=0, choice="auto", parallel=True, nums=[{0}]),
    dict(prompt=1, choice="none", parallel=False, nums=[{0}]),
    dict(prompt=1, choice="none", parallel=True, nums=[{0}]),
    dict(prompt=1, choice="required", parallel=False, nums=[{1}, {0}]),
    dict(prompt=1, choice="required", parallel=True, nums=[{1}, {0}]),
    dict(prompt=1, choice="auto", parallel=False, nums=[{1}, {0}]),
    dict(prompt=1, choice="auto", parallel=True, nums=[{1}, {0}]),
    dict(prompt=2, choice="none", parallel=False, nums=[{0}]),
    dict(prompt=2, choice="none", parallel=True, nums=[{0}]),
    dict(prompt=2, choice="required", parallel=False, nums=[{1}, {1, 2}, {0}]),
    dict(prompt=2, choice="required", parallel=True, nums=[{2}, {0}]),
    dict(prompt=2, choice="auto", parallel=False, nums=[{1}, {1, 2}, {0}]),
    dict(prompt=2, choice="auto", parallel=True, nums=[{2}]),
    dict(prompt=0, choice="auto", parallel=True, nums=[{0}], no_tools=True),
    dict(prompt=0, choice=CHOICE_FUNC, parallel=True, nums=[{1}]),
    dict(prompt=1, choice=CHOICE_FUNC, parallel=True, nums=[{1}]),
    dict(prompt=2, choice=CHOICE_FUNC, parallel=True, nums=[{1}]),
]
CASES = [
    dict(**case, stream=stream)
    for case, stream in itertools.product(CASES, [False, True])
]


def get_temperature(location: str, unit: str):
    if unit == "celsius":
        temp = 30
    elif unit == "fahrenheit":
        temp = 60
    else:
        raise NotImplementedError
    return f"current temperature in {location} is {temp} {unit}"


def get_humidity(location: str):
    if location == "Beijing":
        p = 0.1
    elif location == "Shenzhen":
        p = 0.80
    else:
        p = 0.5
    return f"current get_humidity in {location} is {p}"


tool_functions = {"get_temperature": get_temperature, "get_humidity": get_humidity}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get today's temperature in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or province, e.g. Beijing / Guangdong",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_humidity",
            "description": "Get today's humidity (0-1) in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or province, e.g. Beijing / Guangdong",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

CITIES = ["Beijing", "Shanghai", "Shenzhen", "Tokyo", "New York"]

PROMPTS = [
    "Where is {city}?",
    "What's the temperature in {city}? Do not use Fahrenheit degree.",
    "What's the temperature and humidity in {city}? Do not use Fahrenheit degree.",
]


async def stream_gather(
    response: AsyncStream[ChatCompletionChunk],
) -> ChatCompletionMessage:
    msg = None
    async for chunk in response:
        if msg is None:
            msg = ChatCompletionMessage(role="assistant", tool_calls=[], content="")
            rcontent = ""
        if not chunk.choices:
            continue

        msg.content += chunk.choices[0].delta.content or ""
        rcontent += getattr(chunk.choices[0].delta, "reasoning_content", None) or ""
        for tool_delta in chunk.choices[0].delta.tool_calls or []:
            if tool_delta.index >= len(msg.tool_calls):
                assert len(msg.tool_calls) == tool_delta.index
                msg.tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=tool_delta.id,
                        type="function",
                        function=dict(name="", arguments=""),
                    )
                )
            tool = msg.tool_calls[tool_delta.index]
            tool.function.name += tool_delta.function.name or ""
            tool.function.arguments += tool_delta.function.arguments or ""

    msg.reasoning_content = rcontent
    return msg


async def test(
    client: AsyncOpenAI,
    model: str,
    idx: int,
    /,
    prompt: int,
    choice: str,
    parallel: bool,
    nums: list[set[int]],
    no_tools: bool = None,
    stream: bool = False,
    **kwargs,
):
    tested_nums = []
    for retry in range(MAX_RETRY):
        try:
            logger.info(f"case {idx} begin {retry=}")
            city = CITIES[idx % len(CITIES)]
            prompt_str = PROMPTS[prompt].format(city=city)
            messages = [{"role": "user", "content": prompt_str}]

            real_nums = []
            for i in range(len(nums)):
                kwargs = dict(
                    messages=messages,
                    model=model,
                    tools=TOOLS,
                    tool_choice=choice if i == 0 else "auto",
                    parallel_tool_calls=parallel if i == 0 else True,
                    stream=stream,
                )
                if no_tools:
                    kwargs.pop("tools")
                response: ChatCompletion | AsyncStream[ChatCompletionChunk] = (
                    await client.chat.completions.create(**kwargs)
                )
                if stream:
                    msg = await stream_gather(response)
                else:
                    msg = response.choices[0].message

                content = msg.content or ""
                rcontent = getattr(msg, "reasoning_content", None) or ""
                tool_calls = msg.tool_calls

                logger.info(
                    f"case {idx} round {i}:"
                    f"\n\tkwargs={kwargs}"
                    f"\n\tcontent={repr(content)}"
                    f"\n\trcontent={repr(rcontent)}"
                    f"\n\ttool_calls={tool_calls}"
                )

                messages.append(msg)

                real_nums.append(len(tool_calls) if tool_calls is not None else 0)

                for tool_call in tool_calls or []:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = tool_functions[name](**arguments)
                    messages.append(
                        dict(role="tool", content=result, tool_call_id=tool_call.id)
                    )

                if choice is CHOICE_FUNC:
                    assert (
                        tool_calls
                        and len(tool_calls) == 1
                        and tool_calls[0].function.name == CHOICE_FUNC_NAME
                    )

            logger.info(f"case {idx} end {retry=} real_nums={real_nums}")
            tested_nums.append(real_nums)

            if real_nums[0] not in nums[0]:  # first round must match
                return idx, tested_nums, False
            if all(real_num in num for real_num, num in zip(real_nums, nums)):
                return idx, tested_nums, True
        except:
            logger.exception(f"case {idx} failed {retry=}")
            tested_nums.append(None)
    else:
        return idx, tested_nums, False


async def test_all(ready: asyncio.Event, port: int):
    await asyncio.wait_for(ready.wait(), LAUNCH_TIMEOUT)
    logger.info("test begin")
    client = AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1", api_key="dummy", timeout=REQ_TIMEOUT
    )
    model = (await client.models.list()).data[0].id

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def test_limited(idx, case):
        async with sem:
            return await test(client, model, idx, **case)

    coroutines = [test_limited(idx, case) for idx, case in enumerate(CASES)]
    dones, _ = await asyncio.wait([asyncio.Task(co) for co in coroutines])
    dones = sorted(list(task.result() for task in dones))
    logger.info("all test done")

    all_success = True
    for idx, tested_nums, success in dones:
        logger.info(f"case {idx} {success=} {tested_nums=}: {CASES[idx]}")
    all_success = all_success and success
    return all_success


# ================ test code above, framework code below ==============


async def launch_engine(ready: asyncio.Event):
    if len(sys.argv[1:]) == 1:
        logger.info("skip launch engine")
        ready.set()
        while True:
            await asyncio.sleep(99999)

    logger.info("engine starting...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *sys.argv[1:],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            limit=64 * 2**20,
        )
        asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, proc.terminate)
        while not proc.stdout.at_eof():
            line = await proc.stdout.readline()
            if not line:
                continue
            line = line.decode().rstrip("\n")
            logger_engine.info(line)
            if not ready.is_set() and "Uvicorn running on" in line:
                ready.set()
    finally:
        proc.terminate()
    logger.info("engine stopped!")


def get_port():
    for arg in sys.argv[1:]:
        if arg.startswith("serve.port="):
            return int(arg.split("=")[1])
    raise ValueError("serve.port not found in argv")


async def main():
    port = get_port()
    ready = asyncio.Event()
    engine_task = asyncio.create_task(launch_engine(ready))
    test_task = asyncio.create_task(test_all(ready, port))
    done, _ = await asyncio.wait(
        [engine_task, test_task], return_when="FIRST_COMPLETED"
    )
    for task in done:
        task.result()

    all_success = test_task.result()
    exit(0 if all_success else 1)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        logger.info(
            "Usage1: add this script before engine launch command line\n"
            f"\te.g. python {sys.argv[0]} torchrun ... -m chitu serve.port=PORT ...\n"
            "Usage2: use this script after engine launched\n"
            f"\te.g. python {sys.argv[0]} serve.port=PORT\n"
        )
        exit(1)
    asyncio.run(main())
