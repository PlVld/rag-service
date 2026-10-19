"""Тесты ограничения количества категорий в иерархии."""
from app.api.admin import _apply_category_limit


def _collections(*sizes):
    return [
        {"name": f"coll{i}", "categories": [
            {"category_path": f"C{i} / {j}", "chunk_count": 1}
            for j in range(size)
        ]}
        for i, size in enumerate(sizes)
    ]


def test_no_limit_returns_everything():
    collections = _collections(3, 2)

    meta = _apply_category_limit(collections, None)

    assert meta == {
        "limit": None,
        "truncated": False,
        "total_categories": 5,
        "shown_categories": 5,
    }
    assert [len(c["categories"]) for c in collections] == [3, 2]
    assert all(c["truncated"] is False for c in collections)
    assert [c["total_categories"] for c in collections] == [3, 2]


def test_limit_cuts_first_collection():
    collections = _collections(5, 4)

    meta = _apply_category_limit(collections, 2)

    assert meta["truncated"] is True
    assert meta["total_categories"] == 9
    assert meta["shown_categories"] == 2
    assert len(collections[0]["categories"]) == 2
    assert collections[0]["total_categories"] == 5
    assert collections[0]["truncated"] is True
    assert collections[1]["categories"] == []
    assert collections[1]["total_categories"] == 4
    assert collections[1]["truncated"] is True


def test_limit_spans_collections():
    collections = _collections(2, 4, 3)

    meta = _apply_category_limit(collections, 3)

    assert meta["shown_categories"] == 3
    assert meta["total_categories"] == 9
    assert [len(c["categories"]) for c in collections] == [2, 1, 0]
    assert [c["truncated"] for c in collections] == [False, True, True]


def test_limit_equal_to_total_is_not_truncation():
    collections = _collections(2, 3)

    meta = _apply_category_limit(collections, 5)

    assert meta["truncated"] is False
    assert meta["shown_categories"] == 5
    assert [len(c["categories"]) for c in collections] == [2, 3]
    assert all(c["truncated"] is False for c in collections)


def test_empty_response():
    meta = _apply_category_limit([], 10)

    assert meta == {
        "limit": 10,
        "truncated": False,
        "total_categories": 0,
        "shown_categories": 0,
    }
