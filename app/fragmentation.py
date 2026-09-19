"""IPv4 数据报分片规划核心逻辑（纯标准库实现，无 Web 依赖）。

输入一个 IPv4 数据报的载荷（字节或标准 Base64 字符串）、首部长度
（20～60 字节）、链路 MTU、DF（Don't Fragment）标志，以及可选的原始片
偏移（``original_offset``，以 8 字节为单位）与原始 MF 标志
（``original_more_fragments``）——后两者用于对**已有分片再次切分**。

规划规则：

1. **长度上限**：首部长度 + 载荷长度不能越过 IPv4 数据报 65535 字节上限。
2. **无需分片**：首部加载荷不超过 MTU 时原样输出**一个**分片（偏移与 MF
   继承原始片信息）。
3. **DF 禁止分片**：超过 MTU 且 ``dont_fragment=true`` 时返回
   ``needs_fragmentation``，**不输出任何载荷**。
4. **按序切片**：其余情况下按输入顺序切分——除最终片外，每片载荷取
   「MTU − 首部长度」以内最大的 **8 字节倍数**（IPv4 片偏移以 8 字节计），
   输出偏移以 8 字节为单位。
5. **再次切分**：对已有分片切分时，各子片偏移叠加 ``original_offset``；
   最后子片继承 ``original_more_fragments``，其余子片 MF 恒为 ``true``。
6. **MTU 过小**：MTU 连一个合法非末片载荷（首部之外至少 8 字节）都容不下
   而又必须分片时，抛出 :class:`FragmentationInputError`
   （``error="mtu_too_small"``）。
"""
from __future__ import annotations

import base64

#: 首部长度 / 65535 上限 / 8 字节偏移单位 / Base64 解码复用重组模块定义
from .reassembly import (
    HEADER_MAX,
    HEADER_MIN,
    IPV4_DATAGRAM_MAX,
    OFFSET_UNIT,
    decode_payload,
)

#: MTU 合法性下界：正整数（能否实际分片还取决于首部长度，见 mtu_too_small）
MTU_MIN = 1

#: 规划结论
UNFRAGMENTED = "unfragmented"
FRAGMENTED = "fragmented"
NEEDS_FRAGMENTATION = "needs_fragmentation"

#: 输入错误类型（Web 层映射为 422 detail.error）
ERR_INVALID_HEADER_LENGTH = "invalid_header_length"
ERR_INVALID_MTU = "invalid_mtu"
ERR_INVALID_ORIGINAL_OFFSET = "invalid_original_offset"
ERR_DATAGRAM_TOO_LARGE = "datagram_too_large"
ERR_MTU_TOO_SMALL = "mtu_too_small"


class FragmentationInputError(ValueError):
    """分片规划输入不合法（Web 层映射为 422）。

    - ``error``：错误类型标识（``invalid_header_length`` / ``invalid_mtu``
      / ``invalid_original_offset`` / ``datagram_too_large`` /
      ``mtu_too_small``）；
    - ``limit``：越界错误附带的上限值（总长度类问题为 65535，其余为
      ``None``）；
    - ``mtu`` / ``header_length``：``mtu_too_small`` 时附带的实际 MTU 与
      本次首部长度（其余为 ``None``）。
    """

    def __init__(
        self,
        error: str,
        message: str,
        *,
        limit: int | None = None,
        mtu: int | None = None,
        header_length: int | None = None,
    ) -> None:
        self.error = error
        self.limit = limit
        self.mtu = mtu
        self.header_length = header_length
        super().__init__(message)


def _validate_header_length(header_length: object) -> int:
    """校验首部长度：非 bool 整数且在 20～60 之间。"""
    if isinstance(header_length, bool) or not isinstance(header_length, int):
        raise FragmentationInputError(
            ERR_INVALID_HEADER_LENGTH,
            f"header_bytes 必须是 {HEADER_MIN}～{HEADER_MAX} 的整数: "
            f"{header_length!r}",
        )
    if not HEADER_MIN <= header_length <= HEADER_MAX:
        raise FragmentationInputError(
            ERR_INVALID_HEADER_LENGTH,
            f"header_bytes 必须在 {HEADER_MIN}～{HEADER_MAX} 之间，"
            f"实际 {header_length}",
        )
    return header_length


def _validate_mtu(mtu: object) -> int:
    """校验 MTU：非 bool 正整数。"""
    if isinstance(mtu, bool) or not isinstance(mtu, int):
        raise FragmentationInputError(
            ERR_INVALID_MTU, f"mtu 必须是不小于 {MTU_MIN} 的整数: {mtu!r}"
        )
    if mtu < MTU_MIN:
        raise FragmentationInputError(
            ERR_INVALID_MTU, f"mtu 必须不小于 {MTU_MIN}，实际 {mtu}"
        )
    return mtu


def _validate_original_offset(original_offset: object) -> int:
    """校验原始片偏移：非 bool 非负整数（以 8 字节为单位）。"""
    if isinstance(original_offset, bool) or not isinstance(original_offset, int):
        raise FragmentationInputError(
            ERR_INVALID_ORIGINAL_OFFSET,
            f"original_offset 必须是非负整数: {original_offset!r}",
        )
    if original_offset < 0:
        raise FragmentationInputError(
            ERR_INVALID_ORIGINAL_OFFSET,
            f"original_offset 不能为负: {original_offset}",
        )
    return original_offset


def _validate_flag(value: object, name: str) -> bool:
    """校验布尔标志（DF / 原始 MF）；非布尔抛普通 ValueError。"""
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值 true 或 false: {value!r}")
    return value


def _build_fragment(
    offset: int, more_fragments: bool, piece: bytes, header_length: int
) -> dict:
    """构造单个输出分片（载荷编码为标准 Base64）。"""
    return {
        "offset": offset,
        "more_fragments": more_fragments,
        "payload": base64.b64encode(piece).decode("ascii"),
        "total_length": header_length + len(piece),
    }


def plan_fragmentation(
    header_length: object,
    payload: bytes | str,
    mtu: object,
    dont_fragment: object,
    original_offset: object = 0,
    original_more_fragments: object = False,
) -> dict:
    """按 MTU 与 DF 标志规划一个 IPv4 数据报（或已有分片）的分片结果。

    ``payload`` 为字节串（独立使用时也接受标准 Base64 字符串）；
    ``header_length`` 为 20～60 的整数；``mtu`` 为正整数；``dont_fragment``
    为 DF 布尔标志；``original_offset`` 为以 8 字节为单位的非负原始片偏移，
    ``original_more_fragments`` 为原始片 MF（二者用于对已有分片再次切分）。

    返回可直接 JSON 序列化的字典：

    - 无需分片：``status="unfragmented"``，``fragments`` 为继承原始偏移/MF
      的单个分片；
    - 超过 MTU 且 DF=true：``status="needs_fragmentation"``，
      ``fragment_count=0``、``fragments=[]``（不输出载荷）；
    - 其余完成切片：``status="fragmented"``，``fragments`` 按输入顺序排列，
      除最后一片外每片载荷为 (MTU − 首部长度) 以内最大 8 字节倍数、MF=true；
      最后一片为剩余字节、MF 继承原始 MF；各片偏移均叠加原始片偏移。

    每片给出 ``offset``（8 字节单位）、``more_fragments``、标准 Base64
    ``payload`` 与 ``total_length``（首部 + 本片载荷）。

    语义类输入错误抛出 :class:`FragmentationInputError`（首部/MTU/原始偏移
    非法、首部加载荷越过 65535、MTU 容不下任何合法非末片载荷）；非法
    Base64、标志非布尔等抛出 :class:`ValueError`（Web 层通常已在入口拦截）。
    """
    header_length = _validate_header_length(header_length)
    mtu = _validate_mtu(mtu)
    original_offset = _validate_original_offset(original_offset)
    dont_fragment = _validate_flag(dont_fragment, "dont_fragment")
    original_more_fragments = _validate_flag(
        original_more_fragments, "original_more_fragments"
    )
    if isinstance(payload, str):
        payload = decode_payload(payload)
    if not isinstance(payload, bytes):
        raise ValueError("payload 必须是 Base64 字符串或字节串")

    payload_length = len(payload)
    total_length = header_length + payload_length
    if total_length > IPV4_DATAGRAM_MAX:
        raise FragmentationInputError(
            ERR_DATAGRAM_TOO_LARGE,
            f"数据报长度 {total_length} 越过 IPv4 {IPV4_DATAGRAM_MAX} 字节上限",
            limit=IPV4_DATAGRAM_MAX,
        )

    base = {
        "header_bytes": header_length,
        "mtu": mtu,
        "dont_fragment": dont_fragment,
        "original_offset": original_offset,
    }

    # 无需分片：原样返回一个分片（继承原始片偏移与 MF）
    if total_length <= mtu:
        return {
            **base,
            "status": UNFRAGMENTED,
            "fragment_count": 1,
            "fragments": [
                _build_fragment(
                    original_offset, original_more_fragments, payload, header_length
                )
            ],
        }

    # 超限且 DF=true：通知源端需要分片，不输出载荷
    if dont_fragment:
        return {
            **base,
            "status": NEEDS_FRAGMENTATION,
            "fragment_count": 0,
            "fragments": [],
        }

    # 非末片载荷容量 = MTU - 首部长度，必须放得下一个 8 字节单位
    max_non_final_payload = mtu - header_length
    if max_non_final_payload < OFFSET_UNIT:
        raise FragmentationInputError(
            ERR_MTU_TOO_SMALL,
            f"MTU {mtu} 容纳 {header_length} 字节首部后仅剩 "
            f"{max_non_final_payload} 字节，放不下最小 {OFFSET_UNIT} 字节的"
            f"合法非末片载荷（至少需要 MTU {header_length + OFFSET_UNIT}）",
            mtu=mtu,
            header_length=header_length,
        )

    # 除最终片外，每片载荷取不超过 MTU 的最大 8 字节倍数
    chunk = max_non_final_payload - max_non_final_payload % OFFSET_UNIT

    # 按输入顺序记录每片的（相对字节起点, 载荷）；最后一项为最终片
    pieces: list[tuple[int, bytes]] = []
    pos = 0
    while payload_length - pos > chunk:
        pieces.append((pos, payload[pos : pos + chunk]))
        pos += chunk
    pieces.append((pos, payload[pos:]))  # 最终片：剩余字节（允许不对齐）

    fragments: list[dict] = []
    last_index = len(pieces) - 1
    for index, (piece_pos, piece) in enumerate(pieces):
        more_fragments = True if index < last_index else original_more_fragments
        offset = original_offset + piece_pos // OFFSET_UNIT
        fragments.append(
            _build_fragment(offset, more_fragments, piece, header_length)
        )

    return {
        **base,
        "status": FRAGMENTED,
        "fragment_count": len(fragments),
        "fragments": fragments,
    }
