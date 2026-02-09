import asyncio  # noqa: I001
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from mem0 import AsyncMemory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from nekro_agent.api.core import get_qdrant_config, logger
from nekro_agent.api.schemas import AgentCtx

from .plugin import (
    PluginConfig,
    _last_config_hash,
    _last_dimension,
    _mem0_instance,
    get_memory_config,
    plugin,
)
from .utils import get_model_group_info, get_preset_id

_mem0_lock: asyncio.Lock = asyncio.Lock()

async def reset_qdrant_collection() -> bool:
    """删除并重新创建Qdrant集合，用于维度变更时"""
    try:
        from qdrant_client import AsyncQdrantClient

        qdrant_config = get_qdrant_config()
        collection_name = plugin.get_vector_collection_name()

        # 创建Qdrant客户端
        client = AsyncQdrantClient(
            url=qdrant_config.url,
            api_key=qdrant_config.api_key,
        )

        # 检查集合是否存在
        collections = await client.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections
        )

        if collection_exists:
            # 删除现有集合
            await client.delete_collection(collection_name=collection_name)
            logger.info(f"已删除Qdrant集合: {collection_name}")

        await client.close()
        return True
    except Exception as e:
        logger.error(f"重置Qdrant集合失败: {e}")
        return False

async def create_mem0_client(config: MemoryConfig) -> AsyncMemory:
    # 创建mem0实例
    return AsyncMemory(config)

async def create_mem0_config() -> MemoryConfig:
    # 创建mem0配置实例
    qdrant_config = get_qdrant_config()
    memory_config: PluginConfig = get_memory_config()
    llm_model = get_model_group_info(memory_config.MEMORY_MANAGE_MODEL)
    embedding_model = get_model_group_info(memory_config.TEXT_EMBEDDING_MODEL)

    # 占位符逻辑：当 model 或 openai_base_url 为空时，用 NEED_INPUT 占位，避免底层依赖直接抛错
    NEED_INPUT = "NEED_INPUT"

    def _fallback(value: Optional[str]) -> str:
        return value if (value is not None and str(value).strip() != "") else NEED_INPUT

    llm_model_name = _fallback(llm_model.CHAT_MODEL)
    llm_base_url = _fallback(llm_model.BASE_URL)
    embedder_model_name = _fallback(embedding_model.CHAT_MODEL)
    embedder_base_url = _fallback(embedding_model.BASE_URL)
    return MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "url": qdrant_config.url,
                "api_key": qdrant_config.api_key,
                "collection_name": plugin.get_vector_collection_name(),
                "embedding_model_dims": memory_config.TEXT_EMBEDDING_DIMENSION,
            },
        ),
        llm=LlmConfig(
            provider="openai",
            config={
                "api_key": llm_model.API_KEY,
                "model": llm_model_name,
                "openai_base_url": llm_base_url,
                "temperature": 0,
            },
        ),
        embedder=EmbedderConfig(
            provider="openai",
            config={
                "api_key": embedding_model.API_KEY,
                "model": embedder_model_name,
                "openai_base_url": embedder_base_url,
                "embedding_dims": memory_config.TEXT_EMBEDDING_DIMENSION,
            },
        ),
        version="v1.1",
    )

def _config_incomplete() -> bool:
    """检测插件配置是否完整，若 API_KEY / MODEL / BASE_URL 任何一项为空则判定为不完整。"""
    plugin_cfg: PluginConfig = get_memory_config()
    llm_model = get_model_group_info(plugin_cfg.MEMORY_MANAGE_MODEL)
    embedding_model = get_model_group_info(plugin_cfg.TEXT_EMBEDDING_MODEL)

    def _empty(v: Optional[str]) -> bool:
        return v is None or str(v).strip() == ""

    return any(
        [
            _empty(llm_model.API_KEY),
            _empty(llm_model.CHAT_MODEL),
            _empty(llm_model.BASE_URL),
            _empty(embedding_model.API_KEY),
            _empty(embedding_model.CHAT_MODEL),
            _empty(embedding_model.BASE_URL),
        ],
    )


async def get_mem0_client() -> Optional[AsyncMemory]:
    """异步获取mem0客户端实例"""
    global _mem0_instance, _last_config_hash, _last_dimension

    # 若配置不完整，则跳过初始化，避免底层依赖抛错导致插件加载失败
    if _config_incomplete():
        logger.warning(
            "记忆模块配置不完整：请在插件配置中补齐 记忆管理模型/向量嵌入模型 的 API_KEY/BASE_URL/MODEL。",
        )
        return None

    # 使用原始可序列化字段构建稳定指纹，避免直接哈希模型对象
    plugin_cfg: PluginConfig = get_memory_config()
    qdrant_cfg = get_qdrant_config()
    llm_model = get_model_group_info(plugin_cfg.MEMORY_MANAGE_MODEL)
    embedding_model = get_model_group_info(plugin_cfg.TEXT_EMBEDDING_MODEL)
    collection_name = plugin.get_vector_collection_name()

    current_dimension = plugin_cfg.TEXT_EMBEDDING_DIMENSION

    fingerprint_parts = (
        str(qdrant_cfg.url or ""),
        str(qdrant_cfg.api_key or ""),
        str(collection_name or ""),
        str(plugin_cfg.TEXT_EMBEDDING_DIMENSION),
        str(llm_model.API_KEY or ""),
        str(llm_model.CHAT_MODEL or ""),
        str(llm_model.BASE_URL or ""),
        str(embedding_model.API_KEY or ""),
        str(embedding_model.CHAT_MODEL or ""),
        str(embedding_model.BASE_URL or ""),
    )
    current_hash = hash("|".join(fingerprint_parts))

    # 检测维度是否发生变化
    dimension_changed = (
        _last_dimension is not None and _last_dimension != current_dimension
    )

    # 如果配置变了或者实例不存在，重新初始化（并发保护）
    if _mem0_instance is None or current_hash != _last_config_hash:
        async with _mem0_lock:
            # 双检，避免重复初始化
            if _mem0_instance is None or current_hash != _last_config_hash:
                # 如果维度发生变化，需要重置Qdrant集合
                if dimension_changed:
                    logger.warning(
                        f"检测到嵌入维度从 {_last_dimension} 变更为 {current_dimension}，正在重置Qdrant集合..."
                    )
                    await reset_qdrant_collection()

                memory_config = await create_mem0_config()
                _mem0_instance = await create_mem0_client(memory_config)
                _last_config_hash = current_hash
                _last_dimension = current_dimension
                logger.info("记忆管理器已重新初始化")

    return _mem0_instance


@asynccontextmanager
async def get_memory() -> AsyncIterator[Optional[AsyncMemory]]:
    """官方风格的生命周期管理封装：在 with 块内安全使用内存客户端。"""
    mem = await get_mem0_client()
    try:
        yield mem
    finally:
        pass