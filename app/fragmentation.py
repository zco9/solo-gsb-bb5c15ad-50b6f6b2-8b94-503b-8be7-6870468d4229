"""IPv4 数据报分片规划核心逻辑（纯标准库实现，无 Web 依赖）。

给定 IPv4 首部长度（20～60 字节）、载荷字节、链路 MTU 与 DF 标志，以及
可选的原始片偏移与原始 MF 标志（用于对**已有分片**再次切分），规划该
数据报在此链路上发送时所需的分片序列：

1. **不超 MTU**：数据报（首部 + 载荷）不超过 MTU 时原样返回一个分片
   （``unfragmented``），保留原始片偏移与原始 MF；
2. **超限且 DF=true**：返回 ``needs_fragmentation``，不输出任何分片
   （不输出载荷）；
3. **其余情况切片**（``fragmented``）：除最终片外，每片载荷取不超过
   MTU 的**最大 8 字节倍数**（片偏移以 8 字节计，非末片载荷必须 8 字节
   对齐），最终片取剩余载荷（长度不要求对齐）。各片输出偏移 =
   原始片偏移 + 片内字节偏移 / 8；**最后子片继承原始 MF**，其余子片
   MF=true。

输入约束：首部 + 载荷不得超过 IPv4 数据报 65535 字节上限；需要切片时
MTU 必须能容下一个合法非末片载荷（首部长度 + 至少 8 字节），否则抛出
:class:`FragmentationInputError`（Web 层映射为 422）。
"""
from __future__ import annotations

import base64

from .reassembly import (
    HEADER_MAX,
    HEADER_MIN,
    IPV4_DATAGRAM_MAX,
    OFFSET_UNIT,
    decode_payload,
)

#: MTU 取值下限（正整数，单位字节）
MTU_MIN = 1

#: IPv4 片偏移字段为 13 位，最大值（以 8 字节为单位）
FRAGMENT_OFFSET_MAX = 8191

#: 规划结论
UNFRAGMENTED = "unfragmented"
FRAGMENTED = "fragmented"
NEEDS_FRAGMENTATION = "needs_fragmentation"

#: 输入错误类型（Web 层映射为 422 detail.error）
ERR_INVALID_HEADER_LENGTH = "invalid_header_length"
ERR_INVALID_MTU = "invalid_mtu"
ERR_INVALID_FRAGMENT_OFFSET = "invalid_fragment_offset"
ERR_DATAGRAM_TOO_LARGE = "datagram_too_large"
ERR_MTU_TOO_SMALL = "mtu_too_small"


class FragmentationInputError(ValueError):
    """分片规划输入不合法（Web 层映射为 422）。

    - ``error``：错误类型标识（``invalid_header_length`` / ``invalid_mtu``
      / ``invalid_fragment_offset`` / ``datagram_too_large`` /
      ``mtu_too_small``）；
    - ``limit``：相关阈值——``datagram_too_large`` 为 65535 字节上限，
      ``mtu_too_small`` 为能容下一个合法非末片载荷的最小 MTU
      （首部长度 + 8），其余为 ``None``。
    """

    def __init__(
        self, error: str, message: str, limit: int | None = None
    ) -> None:
        self.error = error
        self.limit = limit
        super().__init__(message)


def _validate_header_length(header_length: object) -> int:
    """校验首部长度：20～60 的整数（拒绝 bool 与浮点等）。"""
    if isinstance(header_length, bool) or not isinstance(header_length, int):
        raise FragmentationInputError(
            ERR_INVALID_HEADER_LENGTH,
            f"header_length 必须是 {HEADER_MIN}～{HEADER_MAX} 的整数: "
            f"{header_length!r}",
        )
    if not HEADER_MIN <= header_length <= HEADER_MAX:
        raise FragmentationInputError(
            ERR_INVALID_HEADER_LENGTH,
            f"header_length 必须在 {HEADER_MIN}～{HEADER_MAX} 之间，"
            f"实际 {header_length}",
        )
    return header_length


def _validate_mtu(mtu: object) -> int:
    """校验 MTU：正整数（拒绝 bool 与浮点等）。"""
    if isinstance(mtu, bool) or not isinstance(mtu, int):
        raise FragmentationInputError(
            ERR_INVALID_MTU,
            f"mtu 必须是不小于 {MTU_MIN} 的整数: {mtu!r}",
        )
    if mtu < MTU_MIN:
        raise FragmentationInputError(
            ERR_INVALID_MTU, f"mtu 必须不小于 {MTU_MIN}，实际 {mtu}"
        )
    return mtu


def _validate_fragment_offset(offset: object) -> int:
    """校验原始片偏移：0～8191 的整数（8 字节单位，拒绝 bool 与浮点等）。"""
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise FragmentationInputError(
            ERR_INVALID_FRAGMENT_OFFSET,
            f"original_offset 必须是 0～{FRAGMENT_OFFSET_MAX} 的整数: {offset!r}",
        )
    if not 0 <= offset <= FRAGMENT_OFFSET_MAX:
        raise FragmentationInputError(
            ERR_INVALID_FRAGMENT_OFFSET,
            f"original_offset 必须在 0～{FRAGMENT_OFFSET_MAX} 之间，"
            f"实际 {offset}",
        )
    return offset


def plan_ipv4_fragmentation(
    header_length: object,
    mtu: object,
    dont_fragment: object,
    payload: object,
    original_offset: object = 0,
    original_more_fragments: object = False,
) -> dict:
    """规划 IPv4 数据报的分片序列，返回可直接 JSON 序列化的结果字典。

    - ``header_length``：IPv4 首部长度（20～60 字节的整数）；
    - ``mtu``：链路 MTU（正整数，单位字节）；
    - ``dont_fragment``：DF 标志（布尔值），``True`` 时禁止分片；
    - ``payload``：载荷字节串（独立使用时也接受标准 Base64 字符串）；
    - ``original_offset``：原始片偏移（8 字节单位，0～8191，默认 0）；
    - ``original_more_fragments``：原始 MF 标志（布尔值，默认
      ``False``）——输入本身是某个数据报的非末片时为 ``True``。

    返回结构（按结论三选一）：

    - ``{"status": "unfragmented", "fragment_count": 1, "fragments": [...]}``：
      数据报不超 MTU，原样返回一个分片（保留原始片偏移与原始 MF）；
    - ``{"status": "needs_fragmentation", "fragment_count": 0,
      "fragments": []}``：超过 MTU 且 DF=true，不输出载荷；
    - ``{"status": "fragmented", "fragment_count": N, "fragments": [...]}``：
      按输入顺序切片后的分片序列（偏移递增）。

    每个分片为 ``{"offset", "more_fragments", "payload", "total_length"}``：
    ``offset`` 以 8 字节计、``payload`` 为标准 Base64、``total_length``
    为首部长度 + 本片载荷字节数（不超过 MTU）。

    语义类输入错误抛出 :class:`FragmentationInputError`（首部长度 / MTU /
    原始片偏移非法、首部加载荷越过 65535、MTU 容不下一个合法非末片载荷）；
    非法 Base64、非布尔标志等抛出 :class:`ValueError`（Web 层通常已在
    入口拦截）。
    """
    header = _validate_header_length(header_length)
    mtu = _validate_mtu(mtu)
    if not isinstance(dont_fragment, bool):
        raise ValueError(f"dont_fragment 必须是布尔值: {dont_fragment!r}")
    offset = _validate_fragment_offset(original_offset)
    if not isinstance(original_more_fragments, bool):
        raise ValueError(
            f"original_more_fragments 必须是布尔值: "
            f"{original_more_fragments!r}"
        )
    if isinstance(payload, str):
        payload = decode_payload(payload)
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("payload 必须是 Base64 字符串或字节串")
    payload = bytes(payload)

    datagram_length = header + len(payload)
    if datagram_length > IPV4_DATAGRAM_MAX:
        raise FragmentationInputError(
            ERR_DATAGRAM_TOO_LARGE,
            f"数据报长度 {datagram_length} 越过 IPv4 "
            f"{IPV4_DATAGRAM_MAX} 字节上限",
            limit=IPV4_DATAGRAM_MAX,
        )

    def build_fragment(
        chunk: bytes, byte_offset: int, more_fragments: bool
    ) -> dict:
        return {
            "offset": offset + byte_offset // OFFSET_UNIT,
            "more_fragments": more_fragments,
            "payload": base64.b64encode(chunk).decode("ascii"),
            "total_length": header + len(chunk),
        }

    # 数据报不超 MTU：原样返回一个分片（保留原始片偏移与原始 MF）
    if datagram_length <= mtu:
        return {
            "status": UNFRAGMENTED,
            "fragment_count": 1,
            "fragments": [build_fragment(payload, 0, original_more_fragments)],
        }

    # 超限且 DF=true：需要分片但被禁止，不输出载荷
    if dont_fragment:
        return {
            "status": NEEDS_FRAGMENTATION,
            "fragment_count": 0,
            "fragments": [],
        }

    # 切片：非末片载荷取不超过 MTU 的最大 8 字节倍数
    max_nonfinal = (mtu - header) // OFFSET_UNIT * OFFSET_UNIT
    if max_nonfinal < OFFSET_UNIT:
        raise FragmentationInputError(
            ERR_MTU_TOO_SMALL,
            f"MTU {mtu} 容不下一个合法非末片载荷（首部 {header} 字节 + "
            f"至少 {OFFSET_UNIT} 字节载荷）",
            limit=header + OFFSET_UNIT,
        )

    fragments: list[dict] = []
    payload_length = len(payload)
    byte_offset = 0
    while byte_offset < payload_length:
        chunk = payload[byte_offset:byte_offset + max_nonfinal]
        chunk_offset = byte_offset
        byte_offset += len(chunk)
        is_last = byte_offset >= payload_length
        # 最后子片继承原始 MF，其余子片 MF=true
        more = original_more_fragments if is_last else True
        fragments.append(build_fragment(chunk, chunk_offset, more))
    return {
        "status": FRAGMENTED,
        "fragment_count": len(fragments),
        "fragments": fragments,
    }
