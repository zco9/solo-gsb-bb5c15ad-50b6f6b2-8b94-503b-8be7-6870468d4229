"""IPv4 数据报分片规划核心逻辑的单元测试。"""
import base64

import pytest

from app.fragmentation import (
    FRAGMENTED,
    NEEDS_FRAGMENTATION,
    UNFRAGMENTED,
    HEADER_MAX,
    HEADER_MIN,
    MTU_MIN,
    OFFSET_UNIT,
    FragmentationInputError,
    plan_fragmentation,
)
from app.reassembly import IPV4_DATAGRAM_MAX, decode_payload, reassemble_ipv4


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def lengths(result):
    return [len(base64.b64decode(f["payload"])) for f in result["fragments"]]


def decode_fragments(result):
    return [
        (f["offset"], f["more_fragments"], decode_payload(f["payload"]))
        for f in result["fragments"]
    ]


# ---------- 无需分片 ----------


def test_small_datagram_returns_single_fragment():
    result = plan_fragmentation(20, b"hello", 1500, False)
    assert result["status"] == UNFRAGMENTED
    assert result["fragment_count"] == 1
    assert result["fragments"] == [
        {"offset": 0, "more_fragments": False, "payload": b64(b"hello"), "total_length": 25}
    ]


def test_exact_mtu_unfragmented():
    # 首部 20 + 载荷 30 = 50，恰好等于 MTU
    result = plan_fragmentation(20, b"A" * 30, 50, False)
    assert result["status"] == UNFRAGMENTED
    assert lengths(result) == [30]
    assert result["fragments"][0]["total_length"] == 50


def test_empty_payload_unfragmented():
    result = plan_fragmentation(20, b"", 20, False)
    assert result["status"] == UNFRAGMENTED
    assert result["fragments"][0] == {
        "offset": 0,
        "more_fragments": False,
        "payload": "",
        "total_length": 20,
    }


def test_unfragmented_inherits_original_offset_and_mf():
    result = plan_fragmentation(20, b"abc", 100, False, 100, True)
    assert result["status"] == UNFRAGMENTED
    assert result["fragments"][0]["offset"] == 100
    assert result["fragments"][0]["more_fragments"] is True


def test_unfragmented_with_df_still_single():
    # 不超 MTU 时 DF 不产生影响
    result = plan_fragmentation(20, b"abc", 1500, True)
    assert result["status"] == UNFRAGMENTED
    assert result["fragment_count"] == 1


def test_echo_fields():
    result = plan_fragmentation(60, b"x", 100, True, 7, True)
    assert result["header_bytes"] == 60
    assert result["mtu"] == 100
    assert result["dont_fragment"] is True
    assert result["original_offset"] == 7


# ---------- DF 需要分片 ----------


def test_oversize_df_needs_fragmentation_no_payload():
    result = plan_fragmentation(20, b"A" * 100, 50, True)
    assert result["status"] == NEEDS_FRAGMENTATION
    assert result["fragment_count"] == 0
    assert result["fragments"] == []


def test_needs_fragmentation_echo_fields():
    result = plan_fragmentation(20, b"A" * 40, 40, True)
    assert result["mtu"] == 40
    assert result["dont_fragment"] is True
    assert result["original_offset"] == 0


# ---------- 常规切片 ----------


def test_basic_fragmentation_chunk_sizes_and_flags():
    result = plan_fragmentation(20, b"A" * 100, 50, False)
    assert result["status"] == FRAGMENTED
    assert result["fragment_count"] == 5
    assert lengths(result) == [24, 24, 24, 24, 4]
    # 非末片 MF=true 且总长不超 MTU，末片 MF=false
    for fragment in result["fragments"][:-1]:
        assert fragment["more_fragments"] is True
        assert fragment["total_length"] == 44
        assert fragment["offset"] % 1 == 0
    last = result["fragments"][-1]
    assert last["more_fragments"] is False
    assert last["total_length"] == 24
    # 偏移以 8 字节计：0, 3, 6, 9, 12
    assert [f["offset"] for f in result["fragments"]] == [0, 3, 6, 9, 12]


def test_chunk_is_largest_multiple_of_8_fitting_mtu():
    # 首部 20、MTU 60：容量 40，本身即 8 的倍数
    r40 = plan_fragmentation(20, b"A" * 100, 60, False)
    assert lengths(r40) == [40, 40, 20]
    assert [f["offset"] for f in r40["fragments"]] == [0, 5, 10]
    # 容量 45 -> 下取整 40；容量 47 -> 40；容量 48 -> 48
    r45 = plan_fragmentation(20, b"A" * 80, 65, False)
    assert lengths(r45) == [40, 40]
    r47 = plan_fragmentation(20, b"A" * 80, 67, False)
    assert lengths(r47) == [40, 40]
    r48 = plan_fragmentation(20, b"A" * 96, 68, False)
    assert lengths(r48) == [48, 48]


def test_sixty_byte_header_respected_in_chunk():
    # 首部 60、MTU 100：容量 40
    result = plan_fragmentation(60, b"A" * 100, 100, False)
    assert lengths(result) == [40, 40, 20]
    for fragment in result["fragments"][:-1]:
        assert fragment["total_length"] == 100


def test_exact_multiple_two_fragments():
    # 载荷 48 恰好 2 个 24：两片，末片 24 字节且 MF=false
    result = plan_fragmentation(20, b"A" * 48, 44, False)
    assert result["fragment_count"] == 2
    assert lengths(result) == [24, 24]
    flags = [f["more_fragments"] for f in result["fragments"]]
    assert flags == [True, False]
    assert [f["offset"] for f in result["fragments"]] == [0, 3]


def test_three_byte_remainder():
    result = plan_fragmentation(20, b"A" * 27, 44, False)
    assert lengths(result) == [24, 3]
    assert [f["offset"] for f in result["fragments"]] == [0, 3]


def test_fragments_concatenate_back_to_payload():
    data = bytes((i * 7 + 3) % 256 for i in range(1000))
    result = plan_fragmentation(20, data, 300, False)
    pieces = b"".join(decode_payload(f["payload"]) for f in result["fragments"])
    assert pieces == data
    # 逐片区间恰好首尾相接（以 8 字节为偏移单位）
    for prev, nxt in zip(result["fragments"], result["fragments"][1:]):
        prev_len = len(base64.b64decode(prev["payload"]))
        assert nxt["offset"] - prev["offset"] == prev_len // OFFSET_UNIT


def test_output_roundtrips_through_reassembly():
    data = bytes((i * 13 + 1) % 256 for i in range(500))
    result = plan_fragmentation(20, data, 64, False)
    reassembled = reassemble_ipv4(20, decode_fragments(result))
    assert reassembled["status"] == "complete"
    assert decode_payload(reassembled["payload"]) == data


# ---------- 对已有分片再次切分 ----------


def test_refragment_offsets_add_original_and_last_inherits_mf():
    # 原始片 offset=10（字节 80）、MF=true、40 字节；MTU 44 -> 24 + 16
    result = plan_fragmentation(20, b"A" * 40, 44, False, 10, True)
    assert result["status"] == FRAGMENTED
    assert decode_fragments(result) == [
        (10, True, b"A" * 24),
        (13, True, b"A" * 16),  # 最后子片继承原始 MF=true
    ]


def test_refragment_last_child_inherits_false_mf():
    # 原始片 MF=false：最后子片 false，其余 true
    result = plan_fragmentation(20, b"A" * 40, 44, False, 10, False)
    assert [f["more_fragments"] for f in result["fragments"]] == [True, False]
    assert [f["offset"] for f in result["fragments"]] == [10, 13]


def test_refragment_offset_stack_is_byte_aligned():
    # 原始偏移非 8 倍数边界（offset 单位 8 字节，叠加点仍以 8 字节计）
    result = plan_fragmentation(20, b"A" * 50, 44, False, 100, False)
    offsets = [f["offset"] for f in result["fragments"]]
    assert offsets == [100, 103, 106]
    assert lengths(result) == [24, 24, 2]
    assert result["fragments"][-1]["more_fragments"] is False
    assert result["fragments"][0]["more_fragments"] is True


def test_refragment_three_children_all_mf_when_original_mf():
    result = plan_fragmentation(20, b"A" * 60, 44, False, 5, True)
    assert lengths(result) == [24, 24, 12]
    assert all(f["more_fragments"] for f in result["fragments"])
    assert [f["offset"] for f in result["fragments"]] == [5, 8, 11]


def test_refragment_single_child_inherits_offset_and_mf_when_fits():
    # 原始片本身不超 MTU：原样一个子片，偏移/MF 全继承
    result = plan_fragmentation(20, b"A" * 10, 44, False, 42, True)
    assert result["status"] == UNFRAGMENTED
    assert decode_fragments(result) == [(42, True, b"A" * 10)]


def test_refragment_df_returns_needs_fragmentation():
    result = plan_fragmentation(20, b"A" * 40, 44, True, 10, True)
    assert result["status"] == NEEDS_FRAGMENTATION
    assert result["fragments"] == []


def test_refragmented_pieces_roundtrip_at_absolute_offsets():
    data = bytes(range(256)) * 4  # 1024 字节原始数据报
    # 先把原始数据报切成大片，再把每个大片用更小 MTU 二次切分
    first_pass = plan_fragmentation(20, data, 200, False)
    assert first_pass["status"] == FRAGMENTED
    all_sub_fragments = []
    for fragment in first_pass["fragments"]:
        sub = plan_fragmentation(
            20,
            decode_payload(fragment["payload"]),
            60,
            False,
            fragment["offset"],
            fragment["more_fragments"],
        )
        all_sub_fragments.extend(decode_fragments(sub))
    # 必须恰好一个 MF=false 末片
    assert sum(1 for _, mf, _ in all_sub_fragments if not mf) == 1
    reassembled = reassemble_ipv4(20, all_sub_fragments)
    assert reassembled["status"] == "complete"
    assert decode_payload(reassembled["payload"]) == data


# ---------- MTU 过小 ----------


def test_mtu_too_small_422_when_fragmentation_required():
    # 首部 20 + 8 = 28 是可分片下限；MTU 27 容不下 8 字节非末片载荷
    with pytest.raises(FragmentationInputError) as exc_info:
        plan_fragmentation(20, b"A" * 100, 27, False)
    error = exc_info.value
    assert error.error == "mtu_too_small"
    assert error.mtu == 27
    assert error.header_length == 20


def test_mtu_equal_header_plus_eight_is_fine():
    # MTU 28：非末片恰好 8 字节
    result = plan_fragmentation(20, b"A" * 20, 28, False)
    assert result["status"] == FRAGMENTED
    assert lengths(result) == [8, 8, 4]
    assert [f["offset"] for f in result["fragments"]] == [0, 1, 2]


def test_mtu_too_small_with_sixty_byte_header():
    with pytest.raises(FragmentationInputError) as exc_info:
        plan_fragmentation(60, b"A" * 100, 67, False)
    assert exc_info.value.error == "mtu_too_small"
    assert exc_info.value.mtu == 67
    assert exc_info.value.header_length == 60
    # MTU 68 恰好可切（8 字节非末片）
    result = plan_fragmentation(60, b"A" * 20, 68, False)
    assert lengths(result) == [8, 8, 4]


def test_mtu_smaller_than_header_but_df_gives_needs_fragmentation():
    # DF=true 时即使 MTU 小到首部都放不下，也优先返回 needs_fragmentation
    result = plan_fragmentation(20, b"A" * 100, 10, True)
    assert result["status"] == NEEDS_FRAGMENTATION


def test_mtu_too_small_not_raised_when_datagram_fits():
    # 单包不超 MTU 时即使 MTU < header+8 也不报错（无需分片）
    result = plan_fragmentation(20, b"", 27, False)
    assert result["status"] == UNFRAGMENTED


# ---------- 65535 上限 ----------


def test_datagram_too_large_raises_before_anything_else():
    payload = b"\x00" * (IPV4_DATAGRAM_MAX - 20 + 1)
    with pytest.raises(FragmentationInputError) as exc_info:
        plan_fragmentation(20, payload, 1500, True)
    error = exc_info.value
    assert error.error == "datagram_too_large"
    assert error.limit == IPV4_DATAGRAM_MAX


def test_datagram_exact_limit_ok():
    payload = b"\x00" * (IPV4_DATAGRAM_MAX - 20)
    result = plan_fragmentation(20, payload, 1500, False)
    assert result["status"] == FRAGMENTED
    assert sum(lengths(result)) == len(payload)


def test_datagram_limit_respects_header_bytes():
    payload = b"\x00" * (IPV4_DATAGRAM_MAX - 60)
    result = plan_fragmentation(60, payload, 1500, False)
    assert result["status"] == FRAGMENTED
    with pytest.raises(FragmentationInputError) as exc_info:
        plan_fragmentation(60, payload + b"\x00", 1500, False)
    assert exc_info.value.error == "datagram_too_large"


# ---------- 输入校验 ----------


def test_invalid_header_length():
    for header in (HEADER_MIN - 1, HEADER_MAX + 1, 0, -1):
        with pytest.raises(FragmentationInputError) as exc_info:
            plan_fragmentation(header, b"abc", 1500, False)
        assert exc_info.value.error == "invalid_header_length"


def test_invalid_header_length_type():
    for header in (20.0, "20", True, None):
        with pytest.raises(FragmentationInputError) as exc_info:
            plan_fragmentation(header, b"abc", 1500, False)
        assert exc_info.value.error == "invalid_header_length"


def test_invalid_mtu():
    for mtu in (0, -1, -100):
        with pytest.raises(FragmentationInputError) as exc_info:
            plan_fragmentation(20, b"abc", mtu, False)
        assert exc_info.value.error == "invalid_mtu"
    assert MTU_MIN == 1


def test_invalid_mtu_type():
    for mtu in (1500.0, "1500", True, None):
        with pytest.raises(FragmentationInputError) as exc_info:
            plan_fragmentation(20, b"abc", mtu, False)
        assert exc_info.value.error == "invalid_mtu"


def test_invalid_original_offset():
    with pytest.raises(FragmentationInputError) as exc_info:
        plan_fragmentation(20, b"abc", 1500, False, -1, False)
    assert exc_info.value.error == "invalid_original_offset"


def test_invalid_original_offset_type():
    for offset in (1.0, "0", True, None):
        with pytest.raises(FragmentationInputError) as exc_info:
            plan_fragmentation(20, b"abc", 1500, False, offset, False)
        assert exc_info.value.error == "invalid_original_offset"


def test_invalid_flag_types():
    with pytest.raises(ValueError):
        plan_fragmentation(20, b"abc", 1500, "false")
    with pytest.raises(ValueError):
        plan_fragmentation(20, b"abc", 1500, False, 0, "true")
    with pytest.raises(ValueError):
        plan_fragmentation(20, b"abc", 1500, 1)


def test_invalid_base64_payload():
    with pytest.raises(ValueError):
        plan_fragmentation(20, "not base64!", 1500, False)


def test_core_accepts_base64_string_payload():
    result = plan_fragmentation(20, "aGVsbG8=", 1500, False)
    assert result["fragments"][0]["payload"] == b64(b"hello")


# ---------- 随机属性 ----------


def test_random_fragmentation_invariants():
    import random

    rng = random.Random(2026)
    for _ in range(300):
        header = rng.randint(HEADER_MIN, HEADER_MAX)
        # 仅构造 IHL 合法的 4 步长首部（核心本身允许任意 20～60 整数）
        data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 3000)))
        mtu = rng.randint(header + 8, header + 500)
        result = plan_fragmentation(header, data, mtu, False)
        if header + len(data) <= mtu:
            assert result["status"] == UNFRAGMENTED
            assert result["fragment_count"] == 1
            continue
        assert result["status"] == FRAGMENTED
        fragments = result["fragments"]
        payloads = [base64.b64decode(f["payload"]) for f in fragments]
        # 非末片：8 字节倍数且总长 <= MTU
        for piece, fragment in zip(payloads[:-1], fragments[:-1]):
            assert len(piece) % OFFSET_UNIT == 0
            assert len(piece) <= mtu - header
            assert (mtu - header) - len(piece) < OFFSET_UNIT
            assert fragment["more_fragments"] is True
        # 末片严格小于一个满非末片容量（否则应并入），且为剩余量
        assert len(payloads[-1]) <= mtu - header
        assert fragments[-1]["more_fragments"] is False
        # 偏移连续、拼接还原
        assert b"".join(payloads) == data
        for prev_f, prev_p, nxt_f in zip(fragments, payloads, fragments[1:]):
            assert nxt_f["offset"] - prev_f["offset"] == len(prev_p) // OFFSET_UNIT


def test_random_refragmentation_invariants():
    import random

    rng = random.Random(99)
    for _ in range(200):
        header = rng.choice([20, 24, 40, 60])
        data = bytes(rng.randrange(256) for _ in range(rng.randint(1, 2000)))
        original_offset = rng.randint(0, 1000)
        original_mf = rng.choice([True, False])
        mtu = rng.randint(header + 8, header + 200)
        result = plan_fragmentation(
            header, data, mtu, False, original_offset, original_mf
        )
        fragments = decode_fragments(result)
        if not fragments:
            continue
        # 首片偏移 == 原始偏移
        assert fragments[0][0] == original_offset
        # 末片 MF 恒等于原始 MF；其余恒 true
        for _, mf, _piece in fragments[:-1]:
            assert mf is True
        assert fragments[-1][1] is original_mf
        # 字节拼接还原（去掉偏移换算后的连续性检查）
        assert b"".join(piece for _, _, piece in fragments) == data
        for (off, _, piece), (next_off, _, _) in zip(fragments, fragments[1:]):
            assert next_off - off == len(piece) // OFFSET_UNIT
