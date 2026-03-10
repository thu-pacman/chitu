# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .utils import (
    get_tool_parser,
    parse_stream_by_parser,
    adjust_message_for_tool_calls,
    patch_chat_template,
)
from .type_def import (
    ChoiceDelta,
    ChoiceToolCall,
    ToolCallParams,
    ToolChoice,
    ToolChoiceNamedTool,
    ToolChoiceFunction,
)

# import is required, otherwise the module is not initialized, and the class will not register
from .qwen3_parser import Qwen3ToolParser
from .deepseekv31_parser import DeepSeekV31ToolParser
from .deepseekv3_parser import DeepSeekV3ToolParser
from .qwen3_coder_parser import Qwen3CoderToolParser
from .qwen3_instruct_parser import Qwen3InstructToolParser
from .glm47_parser import GLM47ToolParser
from .glm45_parser import GLM45ToolParser
from .deepseekv32_parser import DeepSeekV32ToolParser
from .deepseekr1_parser import DeepSeekR1ToolParser
