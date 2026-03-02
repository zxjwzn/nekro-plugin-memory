import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Union

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.config import ModelConfigGroup
from nekro_agent.core.config import config as core_config
from nekro_agent.models.db_chat_channel import DBChatChannel, DefaultPreset

# 在现有import后添加以下代码
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def encode_base62(number: int) -> str:
    """将大整数编码为Base62字符串"""
    if number == 0:
        return BASE62_ALPHABET[0]
    digits = []
    while number > 0:
        number, remainder = divmod(number, 62)
        digits.append(BASE62_ALPHABET[remainder])
    return "".join(reversed(digits))


def decode_base62(encoded: str) -> int:
    """将Base62字符串解码回大整数"""
    number = 0
    for char in encoded:
        number = number * 62 + BASE62_ALPHABET.index(char)
    return number


def encode_id(original_id: str) -> str:
    """将UUID转换为短ID"""
    try:
        uuid_obj = uuid.UUID(original_id)
        return encode_base62(uuid_obj.int)
    except ValueError as err:
        raise ValueError("无效的UUID格式") from err


def decode_id(encoded_id: str) -> str:
    """将短ID转换回原始UUID"""
    try:
        number = decode_base62(encoded_id)
        return str(uuid.UUID(int=number))
    except (ValueError, AttributeError) as err:
        raise ValueError("无效的短ID格式") from err
    
# 根据模型名获取模型组配置项
def get_model_group_info(model_name: str) -> ModelConfigGroup:
    try:
        return core_config.MODEL_GROUPS[model_name]
    except KeyError as e:
        raise ValueError(f"模型组 '{model_name}' 不存在，请确认配置正确") from e

async def get_preset_id(_ctx: AgentCtx) -> str:
    """获取当前人设"""
    # 获取聊天频道
    channel = await DBChatChannel.get_or_none(chat_key=_ctx.chat_key)
    if channel:
        preset = await channel.get_preset()
        if isinstance(preset, DefaultPreset):
            return "default"
        return str(preset.id)
    return "default"