"""Smoke tests: package imports and basic module availability."""


def test_import_recorder():
    """Verify recorder package imports cleanly."""
    import recorder

    assert hasattr(recorder, "__version__")


def test_import_book():
    """Verify book package imports cleanly."""
    import book

    assert hasattr(book, "__version__")


def test_import_features():
    """Verify features package imports cleanly."""
    import features

    assert hasattr(features, "__version__")


def test_import_models():
    """Verify models package imports cleanly."""
    import models

    assert hasattr(models, "__version__")


def test_import_eval():
    """Verify eval package imports cleanly."""
    import eval as eval_pkg

    assert hasattr(eval_pkg, "__version__")


def test_import_explain():
    """Verify explain package imports cleanly."""
    import explain

    assert hasattr(explain, "__version__")


def test_orderbook_basic():
    """Verify OrderBook can be instantiated and basic operations work."""
    from book.orderbook import OrderBook

    ob = OrderBook()
    ob.apply_update(100.0, 1.5, "bid")
    ob.apply_update(101.0, 2.0, "ask")

    assert ob.get_best_bid() == (100.0, 1.5)
    assert ob.get_best_ask() == (101.0, 2.0)
    assert ob.get_mid_price() == 100.5
    assert ob.get_spread() == 1.0

    # Remove level
    ob.apply_update(100.0, 0, "bid")
    assert ob.get_best_bid() == (0.0, 0.0)
