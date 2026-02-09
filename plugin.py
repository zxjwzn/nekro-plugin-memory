from typing import Dict, List, Literal, Optional

from pydantic import Field

from nekro_agent.api.plugin import ConfigBase, NekroPlugin
from nekro_agent.core.core_utils import ExtraField

plugin = NekroPlugin(
    name="记忆模块",
    module_name="nekro_plugin_memory",
    description="让llm拥有长期记忆",
    version="0.2.0",
    author="Zaxpris",
    url="https://github.com/zxjwzn/nekro-plugin-memory",
)

_mem0_instance = None
_last_config_hash = None
_last_dimension = None

@plugin.mount_config()
class PluginConfig(ConfigBase):
    """基础配置"""

    MEMORY_MANAGE_MODEL: str = Field(
        default="default",
        title="记忆管理模型",
        description="用于将传入的记忆内容简化整理的对话模型组",
        json_schema_extra={"ref_model_groups": True, "required": True},
    )
    TEXT_EMBEDDING_MODEL: str = Field(
        default="default",
        title="向量嵌入模型",
        description="用于将传入的记忆进行向量嵌入的嵌入模型组",
        json_schema_extra={"ref_model_groups": True, "required": True},
    )
    TEXT_EMBEDDING_DIMENSION: int = Field(
        default=1024,
        title="嵌入维度",
        description="嵌入维度",
    )
    MEMORY_SEARCH_SCORE_THRESHOLD: float = Field(
        default=0.0,
        title="记忆匹配度阈值",
        description="搜索记忆时，匹配度低于该值的记忆将被过滤掉，取值范围0-1",
    )
    SESSION_ISOLATION: bool = Field(
        default=True,
        title="记忆会话隔离",
        description="开启后bot存储的记忆只对当前会话有效,在其他会话中无法获取",
    )

def get_memory_config() -> PluginConfig:
    """获取最新的记忆模块配置"""
    return plugin.get_config(PluginConfig)