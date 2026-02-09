from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

from .utils import encode_id


def _safe_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default)
    except Exception:
        return default


def _fmt_metadata(md: Optional[Mapping[str, Any]]) -> str:
    if not md:
        return "-"
    try:
        parts = [f"{k}={v}" for k, v in md.items()]
        return ", ".join(parts) if parts else "-"
    except Exception:
        return str(md)


def _coerce_list(data: Any) -> List[Dict[str, Any]]:
    """Coerce mem0 outputs to a list of dict entries.

    Accepts formats like:
    - {"results": [ {...}, ... ]}
    - [ {...}, ... ]
    - { ... } (single entry)
    """
    if data is None:
        return []
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return [x for x in data["results"] if isinstance(x, dict)]
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _filter_by_tags(items: List[Dict[str, Any]], tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter memory items by metadata tags.
    
    Args:
        items: List of memory items to filter
        tags: List of tag values to filter by. If None or empty, returns all items.
        
    Returns:
        Filtered list of memory items
    """
    if not tags:
        return items
    
    filtered_items = []
    for item in items:
        metadata = _safe_get(item, "metadata", {})
        if not isinstance(metadata, dict):
            continue
            
        # Check if any of the requested tags match the item's metadata
        item_type = _safe_get(metadata, "TYPE", "")
        if item_type in tags:
            filtered_items.append(item)
    
    return filtered_items


def _filter_by_score(items: List[Dict[str, Any]], score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Filter memory items by minimum score threshold.
    
    Args:
        items: List of memory items to filter
        score_threshold: Minimum score threshold. If None, returns all items.
        
    Returns:
        Filtered list of memory items with scores above the threshold
    """
    if score_threshold is None:
        return items
    
    filtered_items = []
    for item in items:
        score = _safe_get(item, "score")
        if score is not None:
            try:
                if float(score) >= score_threshold:
                    filtered_items.append(item)
            except (ValueError, TypeError):
                # If score can't be converted to float, include the item
                filtered_items.append(item)
        else:
            # If no score field, include the item
            filtered_items.append(item)
    
    return filtered_items


def _fmt_entry_common(idx: int, item: Mapping[str, Any]) -> List[str]:
    """格式化单条记忆条目（不包含用户信息）"""
    memory = _safe_get(item, "memory", "")
    mid_raw = _safe_get(item, "id", "-")
    try:
        mid = encode_id(mid_raw) if mid_raw and mid_raw != "-" else "-"
    except Exception:
        mid = mid_raw
    created = _safe_get(item, "created_at", "-")
    updated = _safe_get(item, "updated_at", "-")
    tags = _fmt_metadata(_safe_get(item, "metadata"))
    return [
        f"  [{idx}] 记忆: {memory}",
        f"       ID: {mid}",
        f"       创建: {created}    更新: {updated}",
        f"       标签: {tags}",
    ]


def _group_by_user(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """将记忆条目按用户分组"""
    groups = defaultdict(list)
    for item in items:
        user_id = _safe_get(item, "user_id", "-")
        groups[user_id].append(item)
    return dict(groups)


def _format_grouped_memories(items: List[Dict[str, Any]], include_scores: bool = False) -> List[str]:
    """格式化按用户分组的记忆"""
    if not items:
        return []
    
    grouped = _group_by_user(items)
    parts = []
    
    for user_id, user_items in grouped.items():
        # 用户标题
        user_header = f"用户: {user_id} ({len(user_items)} 条记忆)"
        parts.append(user_header)
        
        # 该用户的记忆条目
        for i, item in enumerate(user_items, 1):
            lines = _fmt_entry_common(i, item)
            if include_scores:
                score = _safe_get(item, "score")
                if score is not None:
                    try:
                        lines.insert(1, f"       相关度: {float(score):.4f}")
                    except Exception:
                        lines.insert(1, f"       相关度: {score}")
            parts.append("\n".join(lines))
    
    return parts


def format_search_output(data: Any, tags: Optional[List[str]] = None, score_threshold: Optional[float] = None) -> str:
    """Format mem0.search output into a readable string.
    
    Args:
        data: Raw search results from mem0
        tags: Optional list of tag types to filter by
        score_threshold: Optional minimum score threshold for filtering results
    """
    items = _coerce_list(data)
    items = _filter_by_tags(items, tags)
    items = _filter_by_score(items, score_threshold)
    
    # Build filter description
    filter_parts = []
    if tags:
        filter_parts.append(f"标签: {', '.join(tags)}")
    if score_threshold is not None:
        filter_parts.append(f"阈值: {score_threshold}")
    filter_desc = f"（筛选: {' | '.join(filter_parts)}）" if filter_parts else ""
    
    header = f"搜索结果{filter_desc}（{len(items)} 条）"
    if not items:
        return header + "\n(无结果)"
    
    # 使用分组格式
    parts = [header]
    grouped_parts = _format_grouped_memories(items, include_scores=True)
    parts.extend(grouped_parts)
    
    return "\n\n".join(parts)


def format_get_all_output(data: Any, tags: Optional[List[str]] = None) -> str:
    """Format mem0.get_all output into a readable string.
    
    Args:
        data: Raw get_all results from mem0
        tags: Optional list of tag types to filter by
    """
    items = _coerce_list(data)
    items = _filter_by_tags(items, tags)
    
    tag_filter_desc = f"（筛选: {', '.join(tags)}）" if tags else ""
    header = f"全部记忆{tag_filter_desc}（{len(items)} 条）"
    if not items:
        return header + "\n(无结果)"
    
    # 使用分组格式
    parts = [header]
    grouped_parts = _format_grouped_memories(items, include_scores=False)
    parts.extend(grouped_parts)
    
    return "\n\n".join(parts)


def format_history_output(history: Any) -> str:
    """Format mem0.history output (list of changes)."""
    items = _coerce_list(history)
    header = f"记忆历史（{len(items)} 条）"
    if not items:
        return header + "\n(无记录)"
    parts: List[str] = [header]
    for i, item in enumerate(items, 1):
        action = _safe_get(item, "action", "-")
        created = _safe_get(item, "created_at", "-")
        updated = _safe_get(item, "updated_at", "-")
        is_deleted = _safe_get(item, "is_deleted", 0)
        prev_val = _safe_get(item, "previous_value")
        new_val = _safe_get(item, "new_value")
        mid_raw = _safe_get(item, "memory_id", "-")
        try:
            mid = encode_id(mid_raw) if mid_raw and mid_raw != "-" else "-"
        except Exception:
            mid = mid_raw

        lines = [
            f"[{i}] 动作: {action}    时间: {created}  (更新: {updated})",
            f"     记忆ID: {mid}    删除标记: {is_deleted}",
        ]
        if prev_val is not None and new_val is not None:
            lines.append(f"     变更: {prev_val}  ->  {new_val}")
        elif new_val is not None:
            lines.append(f"     值: {new_val}")
        elif prev_val is not None:
            lines.append(f"     旧值: {prev_val}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def format_single_memory(item: Mapping[str, Any]) -> str:
    """Format mem0.get (single memory) output."""
    user_id = _safe_get(item, "user_id", "-")
    lines = _fmt_entry_common(1, item)
    return f"单条记忆\n用户: {user_id}\n" + "\n".join(lines)


def format_add_output(data: Any) -> str:
    """Format mem0.add output when it returns a list in `results`."""
    items = _coerce_list(data)
    header = f"新增记忆（{len(items)} 条）"
    if not items:
        return header + "\n(无返回)"

    # 对于新增记忆，也使用分组格式
    parts = [header]
    grouped_parts = _format_grouped_memories_with_events(items)
    parts.extend(grouped_parts)

    return "\n\n".join(parts)


def format_delete_output(memory_id: str) -> str:
    """Format mem0.delete output."""
    return f"已删除记忆 ID: {memory_id}"


def _format_grouped_memories_with_events(items: List[Dict[str, Any]]) -> List[str]:
    """格式化包含事件信息的分组记忆（用于新增记忆）"""
    if not items:
        return []
    
    grouped = _group_by_user(items)
    parts = []
    
    for user_id, user_items in grouped.items():
        # 用户标题
        user_header = f"用户: {user_id} ({len(user_items)} 条记忆)"
        parts.append(user_header)
        
        # 该用户的记忆条目
        for i, item in enumerate(user_items, 1):
            lines = _fmt_entry_common(i, item)
            event = _safe_get(item, "event")
            if event:
                lines.insert(1, f"       事件: {event}")
            parts.append("\n".join(lines))
    
    return parts
