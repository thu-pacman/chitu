# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Iterator,
    Literal,
    Sequence,
    TypedDict,
    Mapping,
    Any,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import AutoProcessor
from typing_extensions import NotRequired

from chitu.global_vars import get_global_args

logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str | list[str | dict]
    reasoning_content: NotRequired[str | None]


Dialog = Sequence[Message]


class Processor:
    """Multimodal processor for vision-language models using AutoProcessor."""

    def __init__(self, path: str, trust_remote_code: bool = True):
        """Initialize the processor.

        Args:
            path: Path to the processor config/model
            trust_remote_code: Whether to trust remote code for loading
        """
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=trust_remote_code
        )

    def apply_chat_template(self, dialog, **kwargs):
        return self.processor.apply_chat_template(dialog, **kwargs)

    def __call__(self, **kwargs):
        return self.processor(**kwargs)


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: dict[str, int]

    num_reserved_special_tokens = 256

    def __init__(self, model_path: str, force_full_seq_decode: bool = False):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
            force_full_seq_decode (bool): If true, every time a new token is generated,
                the output string will be re-decoded from all the generated tokens. If
                there are `n` tokens, it will take `O(n^2)` time to decode them all. This
                is requried by some models.
        """
        self.force_full_seq_decode = force_full_seq_decode
        self.decode_cache: dict[int, str] = {}

        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        if "Kimi-K2" in get_global_args().models.name:
            self.pat_str = "|".join(
                [
                    r"""[\p{Han}]+""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""\p{N}{1,3}""",
                    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
                    r"""\s*[\r\n]+""",
                    r"""\s+(?!\S)""",
                    r"""\s+""",
                ]
            )
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|im_end|>",
                "<|im_user|>",
                "<|im_assistant|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
                "<|im_system|>",
                "<|im_middle|>",
            ] + [
                f"<|reserved_special_token_{i}|>"
                for i in range(self.num_reserved_special_tokens - 10)
            ]
        else:
            self.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",  # end of turn
            ] + [
                f"<|reserved_special_token_{i}|>"
                for i in range(5, self.num_reserved_special_tokens - 5)
            ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.debug(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.debug(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str, type(s)

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: list[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int], skip_special_tokens=False) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (list[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """

        if skip_special_tokens:
            raise NotImplementedError(
                "The `Tokenizer` tokenizer backend does not support `skip_special_tokens`"
            )

        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        # Single token decode with cache optimization
        if len(t) == 1 and not self.force_full_seq_decode:
            token_id = t[0]
            if token_id in self.stop_tokens:
                return ""

            if token_id in self.decode_cache:
                return self.decode_cache[token_id]

            text = self.model.decode(cast(list[int], t))
            self.decode_cache[token_id] = text
            return text

        return self.model.decode(cast(list[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> list[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> list[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(
        self,
        dialog: Dialog,
        chat_template_kwargs: Mapping[str, Any] = {},
    ) -> list[int]:
        if chat_template_kwargs:
            raise NotImplementedError(
                "Chat template kwargs are not supported for this tokenizer."
            )
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


def _llama_tokenizer_ignores_tokenizer_json(tokenizer, path: str) -> bool:
    """
    Detect the directory configuration that transformers >= 5.10 resolves to a broken
    `LlamaTokenizer`.

    A directory whose `tokenizer_config.json` declares `LlamaTokenizerFast` but which only
    ships `tokenizer.json` (no `vocab.json`/`merges.txt`) used to fall back to
    `TokenizersBackend`. Since transformers 5.10 that fallback no longer applies to model types
    missing from `TOKENIZER_MAPPING_NAMES` (e.g. `llama`), and `LlamaTokenizer` is returned
    instead. `LlamaTokenizer` rebuilds the BPE from `vocab`/`merges` and ignores
    `tokenizer.json`, which silently drops or mis-splits the prompt text.
    """
    if type(tokenizer).__module__ != "transformers.models.llama.tokenization_llama":
        return False
    return (Path(path) / "tokenizer.json").exists() and not (
        (Path(path) / "vocab.json").exists() or (Path(path) / "merges.txt").exists()
    )


class TokenizerHF:
    def __init__(
        self,
        path: str,
        trust_remote_code: bool = False,
        force_full_seq_decode: bool = False,
        skip_special_tokens: bool = True,
    ):
        self.force_full_seq_decode = force_full_seq_decode
        self.skip_special_tokens = skip_special_tokens
        self.decode_cache: dict[tuple[int, bool], str] = {}

        self.model = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
        )
        if _llama_tokenizer_ignores_tokenizer_json(self.model, path):
            logger.warning(
                f"AutoTokenizer resolved {path} to LlamaTokenizer, which ignores "
                "tokenizer.json; loading PreTrainedTokenizerFast instead"
            )
            self.model = PreTrainedTokenizerFast.from_pretrained(
                path,
                trust_remote_code=trust_remote_code,
            )
        # self.model = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        # Qwen2 don't set bos but have <|im_start|>
        # all special tokens: <|endoftext|> <|im_start|> <|im_end|>
        if "qwen2" in path.lower():
            self.bos_id = self.model.convert_tokens_to_ids("<|im_start|>")
        else:
            self.bos_id = self.model.bos_token_id
        args = get_global_args()
        if hasattr(args, "models") and hasattr(args.models, "eos_token_id"):
            config_eos_tokens = args.models.eos_token_id
        else:
            config_eos_tokens = self.model.eos_token_id

        if isinstance(config_eos_tokens, list):
            self.stop_tokens = set(config_eos_tokens)
            self.eos_id = config_eos_tokens[0]
        elif isinstance(config_eos_tokens, int):
            self.stop_tokens = set([config_eos_tokens])
            self.eos_id = config_eos_tokens
        else:
            self.stop_tokens = set()
            self.eos_id = None

        self.pad_id = self.model.pad_token_id
        self.n_words = self.model.vocab_size

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        try:
            t = self.model.encode(s, add_special_tokens=False)
        except TypeError:
            # Custom tokenizers may use a different kwarg name (e.g. Kimi-K3's
            # TikTokenTokenizer: encode(text, allow_special_tokens=True)).
            t = self.model.encode(s)
        if bos and self.bos_id is not None:
            t.insert(0, self.bos_id)
        if eos and self.eos_id is not None:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int], skip_special_tokens=None) -> str:
        if skip_special_tokens is None:
            skip_special_tokens = self.skip_special_tokens
        if len(t) == 1 and not self.force_full_seq_decode:
            token_id = t[0]
            key = (token_id, skip_special_tokens)
            if key in self.decode_cache:
                return self.decode_cache[key]

            text = self.model.decode(t, skip_special_tokens=skip_special_tokens)
            self.decode_cache[key] = text
            return text

        return self.model.decode(t, skip_special_tokens=skip_special_tokens)


def pin_assistant_reasoning_content(dialog: Dialog) -> list[Message]:
    """Set `reasoning_content` to "" on assistant messages where it is not a string.

    Chat templates in some models (e.g. Qwen3.5, GLM-5) fall back to scan reasoning
    delimiters like `<think>` in `content`, if `reasoning_content` is not an `str`.
    This violates the design that all chat-template-control strings in `content`
    should be rendered as plain-text tokens. Thus we always set `reasoning_content`
    as an `str`.
    """
    new_dialog: list[Message] = []
    for message in dialog:
        if message.get("role") == "assistant" and not isinstance(
            message.get("reasoning_content"), str
        ):
            message = {**message, "reasoning_content": ""}
        new_dialog.append(message)
    return new_dialog


def normalize_dialog(dialog):
    new_dialog = []
    for message in dialog:
        if isinstance(message.get("content"), str):
            new_dialog.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
        else:
            new_dialog.append(message)
    return new_dialog


def _collect_images_from_dialog(dialog, convert_rgb: bool = True):
    """Collect PIL images from dialog messages that contain image content.

    ``convert_rgb=False`` keeps the raw (possibly lazy, possibly RGBA) image so
    the model's own media processor can apply its transparent-background
    compositing (Kimi-K3); ``convert_rgb=True`` matches the historical Qwen-VL
    behavior.
    """
    from PIL import Image

    images = []
    for msg in dialog:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = Image.open(item["image"])
                    images.append(img.convert("RGB") if convert_rgb else img)
    return images


def _extract_vision_outputs(result):
    """Extract (input_ids, pixel_values, grid_thw) from processor output."""
    grid_thw_key = "image_grid_thw" if "image_grid_thw" in result else "grid_thws"
    pv = result["pixel_values"]
    if pv.ndim > 2:
        pv = pv.flatten(1)
    return (
        result["input_ids"].reshape(-1).tolist(),
        pv,
        result[grid_thw_key],
    )


def _encode_kimi_vl(hf_proc, dialog, chat_template_kwargs):
    """Two-step encoding for Kimi K2.5 VL: expand media_pad tokens, then call processor."""
    text = hf_proc.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )

    images = _collect_images_from_dialog(dialog)

    media_pad = "<|media_pad|>"
    for img in images:
        num_tokens = hf_proc.media_processor.media_tokens_calculator(
            {"type": "image", "image": img}
        )
        text = text.replace(media_pad, media_pad * num_tokens, 1)

    medias = [{"type": "image", "image": img} for img in images]
    result = hf_proc(text=[text], medias=medias, return_tensors="pt")
    return _extract_vision_outputs(result)


def _encode_kimi_k3_vl(hf_proc, dialog, chat_template_kwargs):
    """Two-step encoding for Kimi K3 VL.

    K3's chat template renders one ``<|kimi_image_placeholder|>`` per image. The
    processor's ``preprocess_medias`` substitutes a resolution-aware media
    prompt (original WxH, one ``<|media_pad|>``) for each; we then expand every
    pad to the image's post-merge token count so the model can scatter
    per-token features onto the pad positions.
    """
    images = _collect_images_from_dialog(dialog, convert_rgb=False)
    medias = [{"type": "image", "image": img} for img in images]

    _, image_prompts = hf_proc.preprocess_medias(medias)
    text = hf_proc.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=True,
        image_prompts=image_prompts,
        **chat_template_kwargs,
    )

    media_pad = "<|media_pad|>"
    for media in medias:
        num_tokens = hf_proc.media_processor.media_tokens_calculator(media)
        text = text.replace(media_pad, media_pad * num_tokens, 1)

    # The whole rendered text is tokenized with special tokens allowed (the
    # tokenizer's own default), matching the official processor's
    # ``self.tokenizer(text)`` path.
    text_inputs = hf_proc.tokenizer(text, return_tensors="pt")
    preprocessed = hf_proc.media_processor.preprocess(medias, return_tensors="pt")
    return _extract_vision_outputs({**text_inputs, **preprocessed.data})


class ChatFormatHF:
    def __init__(self, tokenizer: TokenizerHF, processor: Processor):
        self.tokenizer = tokenizer
        self.processor = processor

    def encode_header(self, message: Message) -> list[int]:  # ???
        tokens = []
        tokens.extend(self.tokenizer.encode(message["role"], bos=True, eos=False))
        tokens.extend(self.tokenizer.encode("\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> list[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=True)
        )
        tokens.extend(self.tokenizer.encode("\n", bos=False, eos=False))
        return tokens

    def encode_dialog_prompt(
        self,
        dialog: Dialog,
        chat_template_kwargs: Mapping[str, Any] = {},
    ):
        if self.processor:
            normalized = normalize_dialog(dialog)

            # Kimi style (K2.5/K3): the processor's apply_chat_template only
            # tokenizes text, so images need a dedicated two-step encoding.
            hf_proc = self.processor.processor
            if hasattr(hf_proc, "media_processor") and _collect_images_from_dialog(
                normalized, convert_rgb=False
            ):
                # K3's processor renders <|kimi_image_placeholder|> (it defines
                # ``image_placeholder``); K2.5's renders <|media_pad|> directly.
                if hasattr(hf_proc, "image_placeholder"):
                    return _encode_kimi_k3_vl(hf_proc, normalized, chat_template_kwargs)
                return _encode_kimi_vl(hf_proc, normalized, chat_template_kwargs)

            inputs = self.processor.apply_chat_template(
                normalized,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **chat_template_kwargs,
            )

            # Qwen VL style: apply_chat_template returns pixel_values directly.
            if "pixel_values" in inputs:
                return _extract_vision_outputs(inputs)

            return inputs["input_ids"].reshape(-1).tolist()

        if hasattr(self.tokenizer.model, "apply_chat_template"):
            chat_template_kwargs = dict(chat_template_kwargs) or {}

            # `tokenize=True` and `return_dict=False` explicitly request `list[int]`
            # output, even though these are HF transformers' default. This is because
            # models with custom tokenizers such as Kimi-K3's TikTokenTokenizer
            # default to tokenize=False and would return a rendered string instead.
            chat_template_kwargs["add_generation_prompt"] = True
            chat_template_kwargs["tokenize"] = True
            chat_template_kwargs["return_dict"] = False

            try:
                # Try settting `sanitize_control_tokens=True` to encode control tokens
                # appearing in user-supplied text (message content, tools, documents)
                # as plain text, so only the control tokens written by the chat
                # template itself become real token ids.
                ids = self.tokenizer.model.apply_chat_template(
                    dialog, sanitize_control_tokens=True, **chat_template_kwargs
                )
            except ValueError as e:
                if "Can't sanitize" in str(e):
                    # `sanitize_control_tokens=True` may fail if some control tokens
                    # can't be encoded by plain text tokens. This won't happen in well-
                    # defined token sets.
                    logger.warning(
                        f"`sanitize_control_tokens=True` failed when applying chat template. "
                        f"Falling back to applying without it. Cause: {e}."
                    )
                    ids = self.tokenizer.model.apply_chat_template(
                        dialog, **chat_template_kwargs
                    )
                else:
                    raise

            if not isinstance(ids, list):
                raise RuntimeError(
                    "tokenizer.apply_chat_template(tokenize=True) did not return "
                    f"token ids (got {type(ids)}); the chat template of this "
                    "model is incompatible with ChatFormatHF"
                )
            return ids

        else:
            if chat_template_kwargs:
                raise NotImplementedError(
                    "Chat template kwargs are not supported for this tokenizer."
                )
            tokens = []
            for message in dialog:
                tokens.extend(self.encode_message(message))
            tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
            return tokens


from chitu.encoding_dsv32 import encode_messages as encode_messages_dsv32
from chitu.encoding_dsv4 import encode_messages as encode_messages_dsv4
from chitu.output_parser import get_output_parser_params


class ChatFormatHF_dsv32(ChatFormatHF):
    def __init__(self, tokenizer: TokenizerHF, processor: Processor):
        super().__init__(tokenizer, processor)

    def encode_dialog_prompt(
        self,
        messages: Dialog,
        chat_template_kwargs: Mapping[str, Any] = {},
    ):
        tools = chat_template_kwargs.get("tools", None)
        if tools:
            messages = [dict(role="system", tools=tools), *messages]
        enable_thinking = chat_template_kwargs.get("enable_thinking", True)
        enable_thinking = get_output_parser_params(enable_thinking).enable_reasoning
        thinking_mode = "thinking" if enable_thinking else "chat"
        drop_thinking = messages[-1]["role"] == "user"
        prompt = encode_messages_dsv32(
            messages=messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
        )
        return self.tokenizer.encode(prompt, bos=False, eos=False)


class ChatFormatHF_dsv4(ChatFormatHF):
    def __init__(self, tokenizer: TokenizerHF, processor: Processor):
        super().__init__(tokenizer, processor)

    def encode_dialog_prompt(
        self,
        messages: Dialog,
        chat_template_kwargs: Mapping[str, Any] = {},
    ):
        tools = chat_template_kwargs.get("tools", None)
        if tools:
            messages = [dict(role="system", tools=tools), *messages]
        enable_thinking = chat_template_kwargs.get("enable_thinking", True)
        enable_thinking = get_output_parser_params(enable_thinking).enable_reasoning
        thinking_mode = "thinking" if enable_thinking else "chat"
        drop_thinking = messages[-1]["role"] == "user"
        prompt = encode_messages_dsv4(
            messages=messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            reasoning_effort=chat_template_kwargs.get("reasoning_effort", None),
        )
        return self.tokenizer.encode(prompt, bos=False, eos=False)


# Role name mapping: API format -> LLaDA format
_LLADA_ROLE_MAP = {
    "system": "SYSTEM",
    "user": "HUMAN",
    "assistant": "ASSISTANT",
}


class ChatFormatLLaDA(ChatFormatHF):
    """Chat format for LLaDA2.0 models.

    Format: <role>SYSTEM</role>content<|role_end|><role>HUMAN</role>content<|role_end|><role>ASSISTANT</role>
    """

    def __init__(self, tokenizer: TokenizerHF, processor: Processor):
        super().__init__(tokenizer, processor)

    def encode_dialog_prompt(
        self,
        dialog: Dialog,
        chat_template_kwargs: Mapping[str, Any] = {},
    ) -> list[int]:
        enable_thinking = chat_template_kwargs.get("enable_thinking", False)
        thinking_str = (
            "detailed thinking on" if enable_thinking else "detailed thinking off"
        )

        parts: list[str] = []
        has_system = any(m.get("role", "").lower() == "system" for m in dialog)

        for message in dialog:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "".join(text_parts)
            else:
                content = str(content).strip()

            llada_role = _LLADA_ROLE_MAP.get(role.lower(), "HUMAN")
            if role.lower() == "system":
                if not content:
                    content = thinking_str
                else:
                    content = f"{thinking_str}\n{content}"
            parts.append(f"<role>{llada_role}</role>{content}<|role_end|>")

        if not has_system:
            parts.insert(0, f"<role>SYSTEM</role>{thinking_str}<|role_end|>")

        parts.append("<role>ASSISTANT</role>")
        prompt = "".join(parts)
        return self.tokenizer.encode(prompt, bos=False, eos=False)
