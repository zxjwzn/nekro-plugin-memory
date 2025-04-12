import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import httpx
from mem0 import Memory
from pydantic import Field

from nekro_agent.api.core import get_qdrant_config, logger
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.config import ModelConfigGroup
from nekro_agent.core.config import config as core_config
from nekro_agent.services.agent.creator import OpenAIChatMessage
from nekro_agent.services.agent.openai import gen_openai_chat_response
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin, SandboxMethodType

# 扩展元数据
plugin = NekroPlugin(
    name="记忆模块",
    module_name="nekro_plugin_memory",
    description="长期记忆管理系统,支持记忆的增删改查及语义搜索",
    version="0.1.1",
    author="Zaxpris",
    url="https://github.com/zxjwzn/nekro-plugin-memory",
)


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
    
def format_memories(results: List[Dict]) -> str:
    """格式化记忆列表为字符串"""
    if not results:
        return "未找到任何记忆"
    
    formatted = []
    for idx, mem in enumerate(results, 1):
        metadata = mem.get("metadata", {})
        created_at = mem.get("created_at", "未知时间")
        score = mem.get("score", "暂无")
        formatted.append(
        f"{idx}. [ID: {encode_id(mem['id'])}]\n"  # 使用短ID
        f"内容: {mem['memory']}\n"
        f"元数据: {metadata}\n"
        f"创建时间: {created_at}\n"
        f"匹配度: {score}\n",
    )
    return "\n".join(formatted)

#根据模型名获取模型组配置项
def get_model_group_info(model_name: str) -> ModelConfigGroup:
    try:
        return core_config.MODEL_GROUPS[model_name]
    except KeyError as e:
        raise ValueError(f"模型组 '{model_name}' 不存在，请确认配置正确") from e

@plugin.mount_config()
class MemoryConfig(ConfigBase):
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
    SESSION_ISOLATION: bool = Field(
        default=True,
        title="记忆会话隔离",
        description="开启后bot存储的记忆只对当前会话有效,在其他会话中无法获取",
    )
    AUTO_MEMORY_ENABLED: bool = Field(
        default=True,
        title="启用自动记忆检索",
        description="启用后,系统将在对话开始时自动检索与当前对话相关的用户的所有记忆",
    )
    AUTO_MEMORY_SEARCH_LIMIT: int = Field(
        default=5,
        title="自动记忆检索数量上限",
        description="自动检索时返回的记忆条数上限",
    )
    AUTO_MEMORY_CONTEXT_MESSAGE_COUNT: int = Field(
        default=5,
        title="上下文消息数",
        description="可获取到的上下文消息数量",
    )
    AUTO_MEMORY_USE_TOPIC_SEARCH: bool = Field(
        default=False,
        title="启用话题搜索",
        description="启用后,系统将使用LLM来找到最近聊天话题,并通过话题获取相关记忆,可能会延长响应时间",
    )
    TOPIC_CACHE_EXPIRE_SECONDS: int = Field(
        default=60,
        title="话题缓存时长",
        description="系统将临时保留话题,超时后再重新总结",
    )

# 获取最新配置的函数
def get_memory_config() -> MemoryConfig:
    """获取最新的记忆模块配置"""
    return plugin.get_config(MemoryConfig)

# 初始化mem0客户端
_mem0_instance = None
_last_config_hash = None
_thread_pool = ThreadPoolExecutor(max_workers=5)  # 创建一个线程池用于执行同步操作

# 添加记忆注入缓存，避免短时间内重复执行
_memory_inject_cache = {}

# 异步创建Memory实例
async def create_memory_async(config: Dict[str, Any]) -> Memory:
    """将Memory.from_config包装成异步函数，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: Memory.from_config(config),
    )

# 将同步方法包装成异步方法
async def async_mem0_search(mem0, query: str, user_id: str):
    """异步执行mem0.search，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, 
        lambda: mem0.search(query=query, user_id=user_id),
    )

async def async_mem0_get_all(mem0, user_id: str):
    """异步执行mem0.get_all，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, 
        lambda: mem0.get_all(user_id=user_id),
    )

async def async_mem0_add(mem0, messages: str, user_id: str, metadata: Dict[str, Any]):
    """异步执行mem0.add，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, 
        lambda: mem0.add(messages=messages, user_id=user_id, metadata=metadata),
    )

async def async_mem0_update(mem0, memory_id: str, data: str):
    """异步执行mem0.update，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, 
        lambda: mem0.update(memory_id=memory_id, data=data),
    )

async def async_mem0_history(mem0, memory_id: str):
    """异步执行mem0.history，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool, 
        lambda: mem0.history(memory_id=memory_id),
    )

async def get_mem0_client_async():
    """异步获取mem0客户端实例"""
    global _mem0_instance, _last_config_hash
    memory_config = get_memory_config()  # 始终获取最新配置
    qdrant_config = get_qdrant_config()
        
    # 计算当前配置的哈希值
    current_config = {
        "MEMORY_MANAGE_MODEL": memory_config.MEMORY_MANAGE_MODEL,
        "TEXT_EMBEDDING_MODEL": memory_config.TEXT_EMBEDDING_MODEL,
        "TEXT_EMBEDDING_DIMENSION": memory_config.TEXT_EMBEDDING_DIMENSION,
        "llm_model_name": get_model_group_info(memory_config.MEMORY_MANAGE_MODEL).CHAT_MODEL,
        "llm_api_key": get_model_group_info(memory_config.MEMORY_MANAGE_MODEL).API_KEY,
        "llm_base_url": get_model_group_info(memory_config.MEMORY_MANAGE_MODEL).BASE_URL,
        "embedder_model_name": get_model_group_info(memory_config.TEXT_EMBEDDING_MODEL).CHAT_MODEL,
        "embedder_api_key": get_model_group_info(memory_config.TEXT_EMBEDDING_MODEL).API_KEY,
        "embedder_base_url": get_model_group_info(memory_config.TEXT_EMBEDDING_MODEL).BASE_URL,
        "qdrant_url": qdrant_config.url,
        "qdrant_api_key": qdrant_config.api_key,
    }
    
    # 验证字段不能为空字符串
    errors = []
    
    if not current_config["llm_model_name"]:
        errors.append(f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的CHAT_MODEL不能为空")
    if not current_config["llm_api_key"]:
        errors.append(f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的API_KEY不能为空")
    if not current_config["llm_base_url"]:
        errors.append(f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的BASE_URL不能为空")
    if not current_config["embedder_model_name"]:
        errors.append(f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的CHAT_MODEL不能为空")
    if not current_config["embedder_api_key"]:
        errors.append(f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的API_KEY不能为空")
    if not current_config["embedder_base_url"]:
        errors.append(f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的BASE_URL不能为空")
    
    if errors:
        error_message = "记忆模块配置错误：\n" + "\n".join([f"- {error}" for error in errors])
        logger.error(error_message)
        raise ValueError(error_message)
    
    
    current_hash = hash(frozenset(current_config.items()))
    
    # 如果配置变了或者实例不存在，重新初始化
    if _mem0_instance is None or current_hash != _last_config_hash:
        # 重新构建配置
        mem0_client_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": current_config["qdrant_url"],
                    "api_key": current_config["qdrant_api_key"],
                    "collection_name": plugin.get_vector_collection_name(),
                    "embedding_model_dims": current_config["TEXT_EMBEDDING_DIMENSION"],
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": current_config["llm_api_key"],
                    "model": current_config["llm_model_name"],
                    "openai_base_url": current_config["llm_base_url"],
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": current_config["embedder_api_key"],
                    "model": current_config["embedder_model_name"],
                    "openai_base_url": current_config["embedder_base_url"],
                    "embedding_dims": current_config["TEXT_EMBEDDING_DIMENSION"],
                },
            },
            "version": "v1.1",
        }
        
        # 异步创建新实例
        _mem0_instance = await create_memory_async(mem0_client_config)
        _last_config_hash = current_hash
        logger.info("记忆管理器已重新初始化")
        
    return _mem0_instance


@plugin.mount_prompt_inject_method(name="memory_prompt_inject")
async def memory_prompt_inject(_ctx: AgentCtx) -> str:
    """记忆提示注入,在对话开始前检索相关记忆并注入到对话提示中"""
    global _memory_inject_cache
    
    # 没有缓存或缓存已过期，执行正常流程
    memory_config = get_memory_config()
    if not memory_config.AUTO_MEMORY_ENABLED:
        return ""
    
    # 检查缓存是否存在且未过期
    current_time = time.time()
    cache_key = _ctx.from_chat_key
    if cache_key in _memory_inject_cache:
        cache_data = _memory_inject_cache[cache_key]
        if current_time - cache_data["timestamp"] < memory_config.TOPIC_CACHE_EXPIRE_SECONDS:
            logger.info(f"使用缓存的记忆注入结果，剩余有效期：{int(memory_config.TOPIC_CACHE_EXPIRE_SECONDS - (current_time - cache_data['timestamp']))}秒")
            return cache_data["result"]

    try:
        from nekro_agent.models.db_chat_channel import DBChatChannel
        from nekro_agent.models.db_chat_message import DBChatMessage
        
        # 异步获取记忆客户端
        mem0 = await get_mem0_client_async()
        
        # 获取会话信息
        db_chat_channel: DBChatChannel = await DBChatChannel.get_channel(chat_key=_ctx.from_chat_key)
        
        # 从会话键中提取用户ID和类型
        parts = _ctx.from_chat_key.split("_")
        if len(parts) != 2:
            return ""
        
        chat_type, chat_id = parts
        
        # 获取最近消息,用于识别用户和上下文
        record_sta_timestamp = int(time.time() - core_config.AI_CHAT_CONTEXT_EXPIRE_SECONDS)
        recent_messages: List[DBChatMessage] = await (
            DBChatMessage.filter(
                send_timestamp__gte=max(record_sta_timestamp, db_chat_channel.conversation_start_time.timestamp()),
                chat_key=_ctx.from_chat_key,
            )
            .order_by("-send_timestamp")
            .limit(memory_config.AUTO_MEMORY_CONTEXT_MESSAGE_COUNT)
        )
        recent_messages = [msg for msg in recent_messages if msg.sender_bind_qq != "0"] #去除系统发言

        if not recent_messages:
            return ""
        
        # 用于保存找到的用户记忆
        all_memories = []
        
        # 构建上下文内容,用于语义搜索
        context_content = "\\n".join([db_message.parse_chat_history_prompt("") for db_message in recent_messages])
        # 识别参与对话的用户
        user_ids = set()
        
        # 只对私聊启用自动记忆检索
        if chat_type == "private":
            user_ids.add(chat_id)
        elif chat_type == "group":
            # 从最近消息中提取所有发言用户的QQ号
            for msg in recent_messages:
                if msg.sender_bind_qq and msg.sender_bind_qq != "0":
                    user_ids.add(msg.sender_bind_qq)

        # 没有找到有效用户ID,返回空
        if not user_ids:
            return ""
            
        # 将所有用户ID转换为列表，便于后续处理
        user_id_list = list(user_ids)

        # 初始化用于存储LLM分析结果的结构
        user_topics = {user_id: [] for user_id in user_ids}
        user_inferred_memories = {user_id: [] for user_id in user_ids}
        user_candidate_memories = {user_id: [] for user_id in user_ids}
        
        # 使用LLM分析上下文 - 一次性获取所有用户的关键词、推断记忆和待存记忆
        if memory_config.AUTO_MEMORY_USE_TOPIC_SEARCH and context_content:
            try:
                # 获取模型配置
                memory_manage_model_group = get_model_group_info(memory_config.MEMORY_MANAGE_MODEL)
                
                # 准备LLM查询
                system_prompt = """
                你是一个信息分析专家，请基于以下规则，从聊天记录中为每个用户提取信息：
                1. 格式解析要求:
                - 忽略所有包含<|Image:...>的图片消息。
                - 专注处理纯文本消息内容。
                - 保留原始发言顺序。
                2. 信息提取规则：
                (1) 提取核心讨论**话题**：识别每个独立话题（如：游戏、聚餐、工作等），合并同义话题。输出格式: `[话题]:[话题内容]:[涉及的用户id列表]`，例如 `[话题]:露营:[2708583339,987654321]`。
                (2) **推断**用户可能相关的**长期信息**或**兴趣点**：根据对话内容推断，例如用户提到了"喜欢编程"或"下周要去旅游"。输出格式: `[推断记忆]:[推断内容]:[相关用户id]`，例如 `[推断记忆]:对编程感兴趣:[2708583339]`。
                (3) 识别**明确提及**的、适合**长期存储的具体信息**（事实、偏好、约定等）：例如"我的生日是5月1日"、"我们约好周五看电影"。输出格式: `[待存记忆]:[用户id]:[记忆内容]`，例如 `[待存记忆]:123456:生日是5月1日` 或 `[待存记忆]:987654321:周五和123456看电影`。
                3. 输出要求：
                - 每条提取的信息占一行。
                - 严格按照指定的格式输出，包括方括号和冒号。
                - 如果没有提取到某种类型的信息，则不输出该类型的行。

                示例输入:
                [04-10 22:15:22 from_qq:2708583339] 'Zaxpris' 说: 周末去露营怎么样？我最近对户外活动很感兴趣。
                [04-10 22:16:22 from_qq:123456789] 'Tom' 说: <|Image:\\app\\uploads\\xxx.jpg> 好啊，我正好买了新帐篷。
                [04-10 22:17:30 from_qq:987654321] 'Lucy' 说: 露营装备需要准备哪些？我下个月生日是15号，也许可以那时候去？
                [04-10 22:18:00 from_qq:2708583339] 'Zaxpris' 说: 好主意！那我们下个月15号去露营庆祝Lucy生日。

                示例输出:
                [话题]:露营:[2708583339,123456789,987654321]
                [话题]:生日庆祝:[987654321,2708583339]
                [推断记忆]:对户外活动感兴趣:[2708583339]
                [待存记忆]:123456789:购买了新帐篷
                [待存记忆]:987654321:生日是下个月15号
                [待存记忆]:2708583339:约定下个月15号和Lucy、Tom去露营

                需要分析的用户ID列表: {user_id_list_str}
                """
                system_prompt = system_prompt.format(user_id_list_str=", ".join(user_id_list))

                messages = [
                    OpenAIChatMessage.from_text("system", system_prompt),
                    OpenAIChatMessage.from_text("user", context_content),
                ]
                
                # 调用LLM获取分析结果
                llm_response = await gen_openai_chat_response(
                    model=memory_manage_model_group.CHAT_MODEL,
                    messages=[msg.to_dict() for msg in messages],
                    base_url=memory_manage_model_group.BASE_URL,
                    api_key=memory_manage_model_group.API_KEY,
                    stream_mode=False,
                )
                
                # 解析LLM回复
                llm_analysis_output = llm_response.response_content.strip()
                logger.info(f"LLM分析结果:\n{llm_analysis_output}")

                for line in llm_analysis_output.split("\n"):
                    line = line.strip()
                    try:
                        if line.startswith("[话题]:"):
                            parts = line.split(":", 2)
                            if len(parts) == 3:
                                _, topic, user_ids_str = parts
                                topic = topic.strip()
                                user_ids_str = user_ids_str[1:-1] # Remove brackets
                                extracted_user_ids = [uid.strip() for uid in user_ids_str.split(",") if uid.strip()]
                                for user_id in extracted_user_ids:
                                    if user_id in user_topics:
                                        user_topics[user_id].append(topic)
                        elif line.startswith("[推断记忆]:"):
                            parts = line.split(":", 2)
                            if len(parts) == 3:
                                _, inferred_mem, user_ids_str = parts
                                inferred_mem = inferred_mem.strip()
                                user_ids_str = user_ids_str[1:-1] # Remove brackets
                                extracted_user_ids = [uid.strip() for uid in user_ids_str.split(",") if uid.strip()]
                                for user_id in extracted_user_ids:
                                    if user_id in user_inferred_memories:
                                        user_inferred_memories[user_id].append(inferred_mem)
                        elif line.startswith("[待存记忆]:"):
                            parts = line.split(":", 2)
                            if len(parts) == 3:
                                _, user_id, memory_content = parts
                                user_id = user_id.strip()
                                memory_content = memory_content.strip()
                                if user_id in user_candidate_memories:
                                    user_candidate_memories[user_id].append(memory_content)
                    except Exception as e:
                        logger.warning(f"解析LLM分析行失败: '{line}', 错误: {e!s}")
                        continue # 跳过格式错误的行
                
                # 为每个用户进行记忆检索
                for user_id in user_ids:
                    try:
                        # 如果启用会话隔离,添加会话前缀
                        search_user_id = _ctx.from_chat_key + user_id if memory_config.SESSION_ISOLATION else user_id
                        
                        # 合并话题和推断记忆作为搜索查询
                        query_parts = user_topics.get(user_id, []) + user_inferred_memories.get(user_id, [])
                        query = ", ".join(list(set(query_parts))) #去重

                        if query:
                            logger.info(f"用户 {user_id} 的组合搜索查询: {query}")
                            result = await async_mem0_search(
                                mem0,
                                query=query, 
                                user_id=search_user_id,
                            )
                            user_memories = result.get("results", [])
                        else:
                            # 如果没有获取到关键词或推断，获取所有记忆
                            logger.info(f"用户 {user_id} 未获得有效查询内容，获取所有记忆")
                            result = await async_mem0_get_all(mem0, user_id=search_user_id)
                            user_memories = result.get("results", [])
                        
                        # 限制返回记忆数量
                        user_memories = user_memories[:memory_config.AUTO_MEMORY_SEARCH_LIMIT]
                        
                        # 为每个记忆添加用户信息
                        for memory in user_memories:
                            memory["user_qq"] = user_id
                            # 尝试获取用户昵称
                            for msg in recent_messages:
                                if msg.sender_bind_qq == user_id:
                                    memory["user_nickname"] = msg.sender_nickname
                                    break
                            else:
                                memory["user_nickname"] = user_id
                        
                        all_memories.extend(user_memories)
                    except Exception as e:
                        logger.error(f"检索用户 {user_id} 的记忆失败: {e!s}")
                    
            except Exception as e:
                logger.error(f"LLM分析或后续处理失败: {e!s}", exc_info=True)
                # 分析失败时，回退到为每个用户获取所有记忆
                for user_id in user_ids:
                    try:
                        search_user_id = _ctx.from_chat_key + user_id if memory_config.SESSION_ISOLATION else user_id
                        result = await async_mem0_get_all(mem0, user_id=search_user_id)
                        user_memories = result.get("results", [])
                        user_memories = user_memories[:memory_config.AUTO_MEMORY_SEARCH_LIMIT]
                        
                        # 为每个记忆添加用户信息 (代码同上)
                        for memory in user_memories:
                            memory["user_qq"] = user_id
                            for msg in recent_messages:
                                if msg.sender_bind_qq == user_id:
                                    memory["user_nickname"] = msg.sender_nickname
                                    break
                            else:
                                memory["user_nickname"] = user_id
                        
                        all_memories.extend(user_memories)
                    except Exception as inner_e:
                        logger.error(f"回退检索用户 {user_id} 的记忆失败: {inner_e!s}")
        else:
            # 不使用LLM分析时，直接获取所有用户的所有记忆 (逻辑同上)
            for user_id in user_ids:
                try:
                    search_user_id = _ctx.from_chat_key + user_id if memory_config.SESSION_ISOLATION else user_id
                    result = await async_mem0_get_all(mem0, user_id=search_user_id)
                    user_memories = result.get("results", [])
                    user_memories = user_memories[:memory_config.AUTO_MEMORY_SEARCH_LIMIT]
                    
                    # 为每个记忆添加用户信息 (代码同上)
                    for memory in user_memories:
                        memory["user_qq"] = user_id
                        for msg in recent_messages:
                            if msg.sender_bind_qq == user_id:
                                memory["user_nickname"] = msg.sender_nickname
                                break
                        else:
                            memory["user_nickname"] = user_id
                    
                    all_memories.extend(user_memories)
                except Exception as e:
                    logger.error(f"检索用户 {user_id} 的记忆失败: {e!s}")
        
        # 组合最终注入的文本
        memory_text = ""
        
        # 1. LLM 分析结果
        if memory_config.AUTO_MEMORY_USE_TOPIC_SEARCH:
            llm_summary_parts = []
            all_topics = {topic for topics in user_topics.values() for topic in topics}
            all_inferred = {mem for mems in user_inferred_memories.values() for mem in mems}
            
            if all_topics:
                llm_summary_parts.append(f"对话主题: {', '.join(all_topics)}")
            if all_inferred:
                llm_summary_parts.append(f"推断记忆点: {', '.join(all_inferred)}")

            if llm_summary_parts:
                 memory_text += "LLM分析概要:\n" + "\n".join(llm_summary_parts) + "\n-----\n"

        # 2. 检索到的记忆
        if all_memories:
            # 按相关性排序（如果有分数的话）
            all_memories.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
            # 限制返回记忆数量 (二次限制，以防万一)
            all_memories = all_memories[:memory_config.AUTO_MEMORY_SEARCH_LIMIT]
            
            memory_text += "以下是检索到的相关记忆,请优先参考:\n"
            for idx, mem in enumerate(all_memories, 1):
                metadata = mem.get("metadata", {})
                nickname = mem.get("user_nickname", mem.get("user_qq", "未知用户"))
                memory_id = encode_id(mem.get("id","未知ID"))
                score = round(float(mem.get("score", 0)), 3) if mem.get("score") else "暂无"
                memory_text += f"{idx}. [ 记忆归属: {nickname} | 元数据: {metadata} | ID: {memory_id} | 匹配度: {score} ] 内容: {mem['memory']}\n"
            memory_text += "-----\n"
            logger.info(f"找到 {len(all_memories)} 条相关记忆")
        else:
             memory_text += "未检索到与当前对话直接相关的记忆。\n-----\n"

        # 3. 建议存储的新记忆
        candidate_memory_texts = []
        user_nicknames = {} # 缓存昵称避免重复查找
        for msg in recent_messages:
             if msg.sender_bind_qq and msg.sender_bind_qq not in user_nicknames:
                 user_nicknames[msg.sender_bind_qq] = msg.sender_nickname

        for user_id, memories in user_candidate_memories.items():
            if memories:
                nickname = user_nicknames.get(user_id, user_id) # 获取昵称
                for mem_content in memories:
                    candidate_memory_texts.append(f"- {nickname} ({user_id}): {mem_content}")
        
        if candidate_memory_texts:
             memory_text += "LLM建议关注以下信息，可考虑使用`add_memory`存储为长期记忆:\n"
             memory_text += "\n".join(candidate_memory_texts) + "\n"
        
        # 如果没有任何内容，返回空字符串
        if not memory_text.strip():
            return ""

        # 将结果存入缓存
        _memory_inject_cache[cache_key] = {
            "timestamp": current_time,
            "result": memory_text.strip(),
        }
        
        # 清理过期缓存
        expired_keys = [k for k, v in _memory_inject_cache.items() if current_time - v["timestamp"] > memory_config.TOPIC_CACHE_EXPIRE_SECONDS]
        for k in expired_keys:
            del _memory_inject_cache[k]
        
        return memory_text.strip()
    except Exception as e:
        logger.error(f"自动记忆检索或注入失败: {e!s}", exc_info=True)
        return ""  # 出错时返回空，避免中断整个流程

@plugin.mount_sandbox_method(SandboxMethodType.TOOL,name="memory_notice")
async def _memory_notice(_ctx: AgentCtx):
    """
    Do Not Call This Function!
    这是有关记忆模块的提示
    ⚠️ 关键注意：
    - 在使用记忆模块进行记忆存储,搜索等操作时,尽量放在代码最后进行处理,特别是send_msg_text或是send_msg_file
    - user_id必须严格指向记忆的归属主体,metadata中的字段不可替代user_id的作用
    - 如果要存储的记忆中包含时间信息,禁止使用(昨天,前天,之后等)相对时间概念,应使用具体的时间(比如20xx年x月x日 x时x分)
    - 对于虚拟角色,需使用其英文小写全名,例如("hatsune_miku","takanashi_hoshino")
    - 若记忆内容属于对话中的用户,则在存储记忆时user_id=该用户ID(如QQ号为123456的用户说"我的小名是喵喵",则user_id="123456",记忆内容为"小名是喵喵")
    - 若记忆内容属于第三方,则在存储记忆时user_id=第三方ID(如QQ号为123456的用户说"@114514喜欢游泳",则user_id="114514",记忆内容为"喜欢游泳")
    """

@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="添加记忆",
    description="指定用户id并添加长期记忆",
)
async def add_memory(
    _ctx: AgentCtx,
    memory: str,
    user_id: str,
    metadata: Dict[str, Any],
) -> str:
    """添加新记忆到用户档案
    
    Args:
        memory (str): 要添加的记忆内容文本
        **非常重要**
        user_id (str): 关联的用户ID,标识应为用户qq,例如2708583339,而非chat_key.传入空字符串则代表查询有关自身记忆
        metadata (Dict[str, Any]): 元数据标签,{"category": "hobbies"}
        
    Returns:
        str: 记忆ID
        
    Example:
        add_memory("喜欢周末打板球", "114514", {"category": "hobbies","sport_type": "cricket"})
        add_memory("喜欢吃披萨", "123456", {"category": "hobbies","food_type": "pizza"})
        add_memory("喜欢打csgo", "114514", {"category": "hobbies","game_type": "csgo"})
        add_memory("小名是喵喵", "123456", {"category": "name","nickname": "喵喵"})
    """
    memory_config = get_memory_config()
    mem0 = await get_mem0_client_async()
    if user_id == "":
        user_id = core_config.BOT_QQ

    if memory_config.SESSION_ISOLATION :
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")

    try:
        result = await async_mem0_add(mem0, messages=memory, user_id=user_id, metadata=metadata)
        logger.info(f"添加记忆结果: {result}")
        if result.get("results"):
            memory_id = result["results"][0]["id"]
            short_id = encode_id(memory_id)  # 添加编码
            return f"记忆添加成功,ID：{short_id}"
        return ""  # noqa: TRY300
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"添加记忆失败: {e!s}")
        raise RuntimeError(f"记忆添加失败: {e!s}") from e

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="搜索记忆",
    description="通过模糊描述对有关记忆进行搜索",
)
async def search_memory(_ctx: AgentCtx, query: str, user_id: str) -> str:
    """搜索记忆
    在使用该方法前先关注提示词中出现的 "当前会话相关记忆" 字样,如果已有需要的相关记忆,则不需要再使用search_memory进行搜索
    Args:
        query (str): 要搜索的记忆内容文本,可以是问句,例如"喜欢吃什么","生日是多久"
        user_id (str): 要查询的用户唯一标识,标识应为用户qq,例如123456,而非chat_key.传入空字符串则代表查询有关自身记忆
    Examples:
        search_memory("2025年3月1日吃了什么","123456")
    """
    memory_config = get_memory_config()
    mem0 = await get_mem0_client_async()
    if user_id == "":
        user_id = core_config.BOT_QQ
    
    if memory_config.SESSION_ISOLATION :
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")

    try:
        result = await async_mem0_search(mem0, query=query, user_id=user_id)
        logger.info(f"搜索记忆结果: {result}")
        return "以下是你对该用户的记忆:\n" + format_memories(result.get("results", []))
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"搜索记忆失败: {e!s}")
        raise RuntimeError(f"搜索记忆失败: {e!s}") from e

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取记忆",
    description="获取有关该用户的所有记忆",
)
async def get_all_memories( _ctx: AgentCtx,user_id: str) -> str:
    """获取用户所有记忆
    Args:
        user_id (str): 要查询的用户唯一标识,标识应为用户qq,例如123456,而非chat_key.传入空字符串则代表查询有关自身记忆
    Returns:
        str: 格式化后的记忆列表字符串,包含记忆内容和元数据
        
    Example:
        get_all_memories("123456")
    """
    memory_config = get_memory_config()  # 获取最新配置
    mem0 = await get_mem0_client_async()
    if user_id == "":
        user_id = core_config.BOT_QQ

    if memory_config.SESSION_ISOLATION :
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")
    
    try:        
        result = await async_mem0_get_all(mem0, user_id=user_id)
        logger.info(f"获取所有记忆结果: {result}")
        return "以下是你脑海中的记忆:\n" + format_memories(result.get("results", []))
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"获取记忆失败: {e!s}")
        raise RuntimeError(f"获取记忆失败: {e!s}") from e

@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="更新记忆",
    description="根据记忆id更新记忆",
)
async def update_memory(_ctx: AgentCtx,memory_id: str, new_content: str) -> str:
    """更新现有记忆内容
    
    Args:
        memory_id (str): 要更新的记忆ID
        new_content (str): 新的记忆内容文本,至少10个字符
    Returns:
        str: 操作结果状态信息
        
    Example:
        update_memory("bf4d4092...", "喜欢周末打网球")
    """
    mem0 = await get_mem0_client_async()
    try:
        original_id = decode_id(memory_id)  # 解码短ID
    except ValueError as e:
        logger.error(f"无效的记忆ID: {e!s}")
        raise ValueError(f"无效的记忆ID格式: {e!s}") from e
    
    try:        
        result = await async_mem0_update(mem0, memory_id=original_id, data=new_content)
        logger.info(f"更新记忆结果: {result}")
        return result.get("message", "记忆更新成功")
    except httpx.HTTPError as e:
        logger.error(f"更新失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"更新失败: {e!s}")
        raise RuntimeError(f"记忆更新失败: {e!s}") from e

@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="查询记忆修改记录",
    description="查询指定记忆的修改记录",
)
async def get_memory_history( _ctx: AgentCtx, memory_id: str) -> str:
    """获取记忆修改历史记录,可以查询到记忆修改历史
    
    Args:
        memory_id (str): 要查询的记忆ID
    
    Returns:
        str: 格式化后的历史记录字符串,包含记忆修改历史
        
    Example:
        get_memory_history("bf4d4092...")
    """
    mem0 = await get_mem0_client_async()
    try:
        original_id = decode_id(memory_id)  # 解码短ID
    except ValueError as e:
        logger.error(f"无效的记忆ID: {e!s}")
        raise ValueError(f"无效的记忆ID格式: {e!s}") from e
    
    try:
        records = await async_mem0_history(mem0, memory_id=original_id)
        logger.info(f"获取历史记录结果: {records}")
        if not records:
            return "该记忆暂无历史记录"
            
        formatted = []
        for idx, r in enumerate(records, 1):
            formatted.append(
                f"{idx}. [事件更改类型: {r['event']}]\n"
                f"旧内容: {r['old_memory'] or '无'}\n"
                f"新内容: {r['new_memory'] or '无'}\n"
                f"时间: {r['created_at']}\n",
            )
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"获取历史失败: {e!s}")
        raise RuntimeError(f"获取记忆历史记录失败: {e!s}") from e
    
@plugin.mount_cleanup_method()
async def clean_up():
    global _mem0_instance, _last_config_hash, _thread_pool, _memory_inject_cache
    _mem0_instance = None
    _last_config_hash = None
    _thread_pool.shutdown()
    _memory_inject_cache = {}
    """清理插件"""