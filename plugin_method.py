import time
from typing import Any, Dict, List, Optional

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import config as core_config
from nekro_agent.core import logger
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.schemas.chat_message import ChatMessage
from nekro_agent.schemas.signal import MsgSignal
from nekro_agent.services.plugin.base import SandboxMethodType

from .mem0_output_formatter import (
    format_add_output,
    format_delete_output,
    format_get_all_output,
    format_history_output,
    format_search_output,
)
from .mem0_utils import get_mem0_client
from .plugin import PluginConfig, get_memory_config, plugin
from .utils import decode_id, get_preset_id


@plugin.mount_init_method()
async def init_plugin() -> None:
    """初始化插件"""
    global _mem0_instance, _last_config_hash
    await get_mem0_client()


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="添加记忆",
    description="为指定的用户添加记忆",
)
async def add_memory(_ctx: AgentCtx, memory: str, user_id: str, metadata: Dict[str, Any]) -> None:
    """
    Adds a new memory to the user's profile, which is associated with that user.

    Args:
        memory (str): The text content of the memory to be added.
        user_id (str): The associated user ID. This indicates that the memory content is related to the user. This should be the user's ID, not the chat_key.
        metadata (Dict[str, Any]): Metadata tags.
        We support using {TYPE: "TAGS"} to tag different types of memories.
        Currently available memory type tags include:
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        Please refer to the definitions of these tags above for their specific roles.

    Returns:
        None.

    Examples:
        - add_memory("Wants to be called Bob", "user_id", {TYPE: "FACTS"})
        - add_memory("Likes to play games on weekends", "user_id", {TYPE: "PREFERENCES"})
        - add_memory("Has a meeting next Thursday", "user_id", {TYPE: "GOALS"})
        - add_memory("Is an optimistic and friendly person", "user_id", {TYPE: "TRAITS"})
        - add_memory("Is a colleague of 'John Doe'", "user_id", {TYPE: "RELATIONSHIPS"})
        - add_memory("Attended a wedding last month", "user_id", {TYPE: "EVENTS"})
        - add_memory("Mentioned their views on life", "user_id", {TYPE: "TOPICS"})
    """

    # 提示词原文
    """
    为用户的个人资料添加一条新记忆,添加的记忆与该用户相关.

    Args:
        memory (str): 要添加的记忆的文本内容.
        user_id (str): 关联的用户ID.代表添加的记忆内容于用户相关,这应该是用户的ID，而不是chat_key.
        metadata (Dict[str, Any]): 元数据标签.
        我们支持使用{TYPE: "TAGS"}来对不同类型的记忆进行标记
        目前可用的记忆类型标签包括：
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        各个标签的具体作用请参考上文我们对这些标签的定义
    Returns:
        None.
    
    Examples
        - add_memory("希望被称呼为Bob", "user_id", {TYPE: "FACTS","CONFIDENCE": "VERY_HIGH"})
        - add_memory("喜欢在周末玩游戏", "user_id", {TYPE: "PREFERENCES","CONFIDENCE": "MEDIUM"})
        - add_memory("下周四有会议开展", "user_id", {TYPE: "GOALS","CONFIDENCE": "LOW"})
        - add_memory("是个乐观友善的人", "user_id", {TYPE: "TRAITS","CONFIDENCE": "VERY_HIGH"})
        - add_memory("和'张三'是同事", "user_id", {TYPE: "RELATIONSHIPS","CONFIDENCE": "HIGH"})
        - add_memory("上个月参加了婚礼", "user_id", {TYPE: "EVENTS","CONFIDENCE": "VERY_HIGH"})
        - add_memory("有提到对于人生的看法", "user_id", {TYPE: "EVENTS","CONFIDENCE": "HIGH"})
    """
    mem0 = await get_mem0_client()
    plugin_config: PluginConfig = get_memory_config()
    if not mem0:
        logger.error("无法获取 mem0 客户端实例，无法添加记忆")
        return

    # 仅在有 chat_key 且启用隔离时设置 run_id
    run_id = None
    if plugin_config.SESSION_ISOLATION and _ctx.chat_key:
        run_id = str(_ctx.chat_key)

    # 不使用try except,出现问题直接报错
    res = await mem0.add(
        memory,
        user_id=user_id,
        agent_id=await get_preset_id(_ctx),
        run_id=run_id,
        metadata=metadata,
    )
    msg = format_add_output(res)
    logger.info(msg)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="搜索记忆",
    description="通过自然语言问句进行记忆查询，支持按标签过滤",
)
async def search_memory(_ctx: AgentCtx, query: str, user_id: str, tags: Optional[List[str]] = None) -> str:
    """
    Retrieves relevant memories for a specified user using a natural language query.
    When the required relevant memories do not appear in the context, you can try using this method to search.

    Args:
        query (str): The query string, which can be a natural language question or keywords.
        user_id (str): The associated user ID. This indicates the query is related to the user. It should be the user's ID, not the chat_key.
        tags (Optional[List[str]]): An optional list of memory type tags to filter by.
        Currently available memory type tags include:
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        Please refer to the definitions of these tags above for their specific roles.

    Returns:
        str: A structured text of search results, suitable for direct display; returns an error message on failure.

    Examples:
        search_memory("What does he like to eat?", "17295800")
        search_memory("Topics discussed last week", "73235808", ["TOPICS"])
        search_memory("His personal preferences", "12345", ["PREFERENCES", "TRAITS"])
    """

    # 提示词原文
    """
    通过自然语言问句检索指定用户的相关记忆.
    当上下文中没有出现所需要的相关记忆时，可以尝试使用此方法进行搜索
    Args:
        query (str): 查询语句，自然语言问题或关键词.
        user_id (str): 关联的用户ID。代表查询的记忆与该用户相关,这应该是用户的ID,而不是 chat_key。
        tags (Optional[List[str]]): 可选的记忆类型标签过滤列表。
        目前可用的记忆类型标签包括：
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        各个标签的具体作用请参考上文我们对这些标签的定义

    Returns:
        str: 结构化的搜索结果文本，适合直接展示；发生异常时返回报错信息

    Examples:
        search_memory("他喜欢吃什么？", "17295800")
        search_memory("上周聊过的话题", "73235808", ["TOPICS"])
        search_memory("他的个人喜好", "12345", ["PREFERENCES", "TRAITS"])
    """
    mem0 = await get_mem0_client()
    plugin_config: PluginConfig = get_memory_config()
    if not mem0:
        logger.error("无法获取 mem0 客户端实例，无法添加记忆")
        return "无法获取 mem0 客户端实例，无法添加记忆，请检查插件配置是否正确"

    # 仅在有 chat_key 且启用隔离时设置 run_id
    run_id = None
    if plugin_config.SESSION_ISOLATION and _ctx.chat_key:
        run_id = str(_ctx.chat_key)

    results = await mem0.search(
        query,
        user_id=user_id,
        agent_id=await get_preset_id(_ctx),
        run_id=run_id,
    )
    return format_search_output(results, tags, plugin_config.MEMORY_SEARCH_SCORE_THRESHOLD)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取所有记忆",
    description="获取指定用户的所有记忆，支持按标签过滤",
)
async def get_all_memory(_ctx: AgentCtx, user_id: str, tags: Optional[List[str]] = None) -> str:
    """
    Gets all memory entries for a specified user.

    Args:
        user_id (str): The associated user ID. This indicates the memories to be retrieved are related to the user. It should be the user's ID, not the chat_key.
        tags (Optional[List[str]]): An optional list of memory type tags to filter by.
        Currently available memory type tags include:
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        Please refer to the definitions of these tags above for their specific roles.

    Returns:
        str: A structured text of the memory list; returns an empty string on failure.

    Examples:
        get_all_memory("17295800")
        get_all_memory("", ["PREFERENCES"])
        get_all_memory("12345", ["FACTS", "RELATIONSHIPS"])
    """

    # 提示词原文
    """
    获取指定用户的全部记忆条目.

    Args:
        user_id (str): 关联的用户ID.代表获取的记忆与该用户相关，这应该是用户的ID，而不是 chat_key。
        tags (Optional[List[str]]): 可选的记忆类型标签过滤列表。
        目前可用的记忆类型标签包括：
            FACTS, PREFERENCES, GOALS, TRAITS, RELATIONSHIPS, EVENTS, TOPICS
        各个标签的具体作用请参考上文我们对这些标签的定义

    Returns:
        str: 结构化的记忆列表文本；发生异常时返回空字符串。

    Examples:
        get_all_memory("17295800")
        get_all_memory("", ["PREFERENCES"])
        get_all_memory("12345", ["FACTS", "RELATIONSHIPS"])
    """
    mem0 = await get_mem0_client()
    plugin_config: PluginConfig = get_memory_config()
    if not mem0:
        logger.error("无法获取 mem0 客户端实例，无法添加记忆")
        return "无法获取 mem0 客户端实例，无法添加记忆，请检查插件配置是否正确"

    # 仅在有 chat_key 且启用隔离时设置 run_id
    run_id = None
    if plugin_config.SESSION_ISOLATION and _ctx.chat_key:
        run_id = str(_ctx.chat_key)

    results = await mem0.get_all(
        user_id=user_id,
        agent_id=await get_preset_id(_ctx),
        run_id=run_id,
    )
    return format_get_all_output(results, tags)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取记忆历史",
    description="获取指定用户的记忆历史",
)
async def get_memory_history(_ctx: AgentCtx, memory_id: str) -> str:
    """
    Gets the version/change history of a memory.
    When you want to know the specific change process of a memory, you can use this method.

    Args:
        memory_id (str): The unique ID of the memory.

    Returns:
        str: A structured text of the history; returns a specific error message on failure.

    Example:
        get_memory_history("01J5ZQ1A8S3J6M9Y4K2B7N")
        get_memory_history("3JSK76D9B837N")
    """

    # 提示词原文
    """
    获取某条记忆的版本/变更历史.
    当你想知道某条记忆的具体变更过程时，可以使用此方法。

    Args:
        memory_id (str): 记忆的唯一ID

    Returns:
        str: 结构化的历史记录文本；发生异常时返回具体报错信息.

    Example:
        get_memory_history("01J5ZQ1A8S3J6M9Y4K2B7N")
        get_memory_history("3JSK76D9B837N")
    """
    mem0 = await get_mem0_client()
    if not mem0:
        logger.error("无法获取 mem0 客户端实例，无法添加记忆")
        return "无法获取 mem0 客户端实例，无法添加记忆，请检查插件配置是否正确"

    # 支持传入 Base62 短ID，优先尝试还原为 UUID
    try:
        raw_id = decode_id(memory_id)
    except Exception:
        raw_id = memory_id

    results = await mem0.history(
        memory_id=raw_id,
    )
    return format_history_output(results)


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="删除记忆",
    description="删除指定ID的记忆",
)
async def delete_memory(_ctx: AgentCtx, memory_id: str) -> None:
    """
    Deletes a specific memory by its ID.

    Args:
        memory_id (str): The unique ID of the memory to delete.

    Returns:
        None.

    Examples:
        delete_memory("01J5ZQ1A8S3J6M9Y4K2B7N")
        delete_memory("3JSK76D9B837N")
    """

    # 提示词原文
    """
    删除指定ID的记忆.

    Args:
        memory_id (str): 要删除的记忆的唯一ID

    Returns:
        None.

    Examples:
        delete_memory("01J5ZQ1A8S3J6M9Y4K2B7N")
        delete_memory("3JSK76D9B837N")
    """
    mem0 = await get_mem0_client()
    if not mem0:
        logger.error("无法获取 mem0 客户端实例，无法删除记忆")
        return

    # 支持传入 Base62 短ID，优先尝试还原为 UUID
    try:
        raw_id = decode_id(memory_id)
    except Exception:
        raw_id = memory_id

    await mem0.delete(memory_id=raw_id)
    msg = format_delete_output(memory_id)
    logger.info(msg)


# @plugin.mount_sandbox_method(
#    SandboxMethodType.BEHAVIOR,
#    name="删除所有记忆",
#    description="删除指定用户的所有记忆",
# )
async def delete_all_memory(_ctx: AgentCtx, user_id: str) -> None:
    """
    Deletes all memories for a specified user.

    Args:
        user_id (str): The associated user ID. This indicates that the memories to be deleted are related to the user. It should be the user's ID, not the chat_key.
    Returns:
        None.

    Example:
        delete_all_memory("17295800")
        delete_all_memory("")
    """

    # 提示词原文
    """
    删除指定用户的所有记忆.

    Args:
        user_id (str): 关联的用户ID。代表要删除的记忆与该用户相关，这应该是用户的ID，而不是 chat_key。
    Returns:
        None.

    Example:
        delete_all_memory("17295800")
        delete_all_memory("")
    """
    mem0 = await get_mem0_client()
    plugin_config: PluginConfig = get_memory_config()
    if not mem0:
        return

    # 仅在有 chat_key 且启用隔离时设置 run_id
    run_id = None
    if plugin_config.SESSION_ISOLATION and _ctx.chat_key:
        run_id = str(_ctx.chat_key)

    await mem0.delete_all(
        user_id=user_id,
        agent_id=await get_preset_id(_ctx),
        run_id=run_id,
    )


async def reset_memory_command(_ctx: AgentCtx, chatmessage: ChatMessage, args: str):  # noqa: ARG001
    try:
        await delete_all_memory(_ctx, chatmessage.sender_id)
        await _ctx.ms.send_text(_ctx.chat_key, "已清空记忆", _ctx)
    except Exception:
        logger.error("清空记忆失败")


COMMAND_MAP = {
    "del_all_mem": reset_memory_command,
}


@plugin.mount_on_user_message()
async def on_message(_ctx: AgentCtx, chatmessage: ChatMessage) -> MsgSignal:
    msg_text = chatmessage.content_text.strip()

    # 检测是否是指令（以/开头）
    if not msg_text.startswith("/"):
        return MsgSignal.CONTINUE

    # 解析指令和参数
    parts = msg_text[1:].split()  # 移除/前缀并分割
    if not parts:
        return MsgSignal.CONTINUE

    command = parts[0].lower()
    args = " ".join(parts[1:]) if len(parts) > 1 else ""

    # 检查指令是否存在于指令映射中
    if command in COMMAND_MAP:
        try:
            # 执行对应的指令处理函数
            await COMMAND_MAP[command](_ctx, chatmessage, args)
            logger.info(f"成功执行指令: {command}")
        except Exception as e:
            logger.error(f"执行指令 {command} 时发生错误: {e}")
        return MsgSignal.BLOCK_ALL

    # 未知指令，继续处理
    return MsgSignal.CONTINUE


@plugin.mount_prompt_inject_method(name="nekro_plugin_memory_prompt_inject")
async def inject_memory_prompt(_ctx: AgentCtx) -> str:
    db_chat_channel: DBChatChannel = await DBChatChannel.get_channel(
        chat_key=_ctx.chat_key,
    )  
    
    # 获取最近消息,用于识别用户和上下文
    record_sta_timestamp = int(
        time.time() - core_config.AI_CHAT_CONTEXT_EXPIRE_SECONDS,
    )
    recent_messages: List[DBChatMessage] = await (
        DBChatMessage.filter(
            send_timestamp__gte=max(
                record_sta_timestamp,
                db_chat_channel.conversation_start_time.timestamp(),
            ),
            chat_key=_ctx.from_chat_key,
        )
        .order_by("-send_timestamp")
        .limit(core_config.AI_CHAT_CONTEXT_MAX_LENGTH)
    )
    recent_messages = [
        msg for msg in recent_messages if msg.sender_id != "0" and msg.sender_id != "-1"
    ]  # 去除系统发言

    #获取所有发言用户
    user_ids = set()

    for msg in recent_messages:
        if msg.sender_id and msg.sender_id != "0" and msg.sender_id != "-1":
            user_ids.add(msg.sender_id)
    
    user_id_list = list(user_ids)

    memory_context = "预搜索结果为空"
    if user_id_list:
        memory_context = ""
        for uid in user_id_list:
            try:
                mem_text = await get_all_memory(_ctx, uid, ["FACTS","TRAITS","RELATIONSHIPS"])
                if "(无结果)" in mem_text:
                    continue
                memory_context += f"{mem_text}\n"
                logger.info(f"为用户 {uid} 注入记忆:\n{mem_text}")
            except Exception as e:
                logger.error(f"获取用户 {uid} 记忆失败: {e}")
        
        if not memory_context.strip():
            memory_context = "预搜索结果为空"
    
    PROMPT = f"""
    This is a plugin for memory management. You can use it to store and retrieve memory information related to users.
    We have defined the following metadata tags to help you categorize different memories:
    TYPE:
        - FACTS: For factual information that is unlikely to change in the short term, such as name, birthday, occupation, etc.
        - PREFERENCES: For user's personal preferences, such as "likes classical music", "hates cilantro".
        - GOALS: For user's goals or aspirations, such as "wants to learn Python by the end of the year", "plans to travel to Japan".
        - TRAITS: For describing a user's personality or habits, such as "is an optimistic person", "has a habit of morning jogging".
        - RELATIONSHIPS: For recording user's interpersonal relationships, such as "is a colleague of 'John Doe'", "pet cat is named 'Mimi'".
        - EVENTS: For recording events or milestones, such as "attended a wedding last month", "completed a marathon last year".
        - TOPICS: For recording topics the user has discussed, such as "has talked about artificial intelligence", "has talked about relationships".
    CONFIDENCE:
        - VERY_HIGH: Almost certainly a fact, with conclusive evidence or explicitly confirmed by the user.
        - HIGH: Supported by strong evidence, very likely to be correct.
        - MEDIUM: Some evidence supports it, but further verification is needed.
        - LOW: Unlikely, but still a small possibility.
        - VERY_LOW: Pure speculation or has been proven false.

    Here are some things to keep in mind when using the memory plugin:
    When performing memory access operations with the memory module (e.g., add_memory, search_memory), the best practice is to handle these operations at the end of your code, especially after calling functions like send_msg_text or send_msg_file.
    Actively analyze chat history to extract information about the user, store it using add_memory, and categorize it using the various metadata tags above.

    Here is the memory you might use in the conversation:
    {memory_context}
    Due to the limited pre-search functionality, if the above does not contain the memory you need, you can try calling search_memory to find it.
    """

    # 提示词原文
    """
    这是一个用于进行记忆管理的插件,你可以通过它来存储和检索与用户相关的记忆信息.
    我们规定了以下几个元数据标签,方便你对不同的记忆进行分类
    TYPE:
        - FACTS: 适用于短期内不会改变的事实信息，例如姓名、生日、职业等.
        - PREFERENCES: 适用于用户的个人喜好，例如“喜欢古典音乐”、“讨厌吃香菜”.
        - GOALS: 适用于用户的目标或愿望，例如“想在年底前学会Python”、“计划去日本旅游”.
        - TRAITS: 适用于描述用户的性格或习惯，例如“是个乐观的人”、“有晨跑的习惯”.
        - RELATIONSHIPS: 适用于记录用户的人际关系，例如“和‘张三’是同事”、“宠物猫叫‘咪咪’”.
        - EVENTS: 适用于记录事件或里程碑，例如“上个月参加了婚礼”、“去年完成了马拉松”.
        - TOPICS: 适用于记录用户曾聊过的话题，例如“有聊过人工智能”、“曾聊过恋爱话题”.
    CONFIDENCE:
        - VERY_HIGH: 几乎可以肯定是事实，有确凿证据或由用户明确确认
        - HIGH: 有很强的证据支持，大概率是正确的
        - MEDIUM: 有一定的证据支持，但仍需进一步验证
        - LOW: 不太可能，但仍有微小可能性
        - VERY_LOW: 纯属猜测或已被证伪

    以下是在使用记忆插件时需要注意的内容:
    在使用记忆模块进行记忆存取操作时（例如 add_memory、search_memory），最佳实践是将这些操作放在代码的末尾处理，尤其是在调用 send_msg_text 或 send_msg_file 等函数之后
    在聊天中主动分析聊天记录来提取有关用户的信息,使用add_memory进行存储并使用以上各种元数据标签进行分类
    """
    return PROMPT  # noqa: RET504


@plugin.mount_cleanup_method()
async def clean_up() -> None:
    """清理插件"""
    global _mem0_instance, _last_config_hash, _memory_inject_cache
    _mem0_instance = None
    _last_config_hash = None
    _memory_inject_cache = {}
