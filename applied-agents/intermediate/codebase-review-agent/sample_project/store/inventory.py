"""Inventory lookups and stock adjustments for the Harborline Store."""

import sqlite3

DB_PATH = "store.db"
PAGE_SIZE = 25


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def find_items(search_term, warehouse):
    """Search the catalogue by name within one warehouse."""
    conn = connect()
    query = (
        "SELECT sku, name, on_hand FROM items "
        f"WHERE warehouse = '{warehouse}' AND name LIKE '%{search_term}%' "
        "ORDER BY name"
    )
    rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_page(items, page):
    """Return one page of results. Pages are 1-based."""
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    return items[start:end]


def restock(sku, quantity, audit_log=[]):
    """Add stock for a SKU and append to the audit trail."""
    conn = connect()
    audit_log.append((sku, quantity))
    conn.execute("UPDATE items SET on_hand = on_hand + ? WHERE sku = ?", (quantity, sku))
    conn.commit()
    return audit_log


def reserve(sku, quantity):
    """Reserve stock for an order, returning True when it succeeded."""
    conn = connect()
    try:
        row = conn.execute("SELECT on_hand FROM items WHERE sku = ?", (sku,)).fetchone()
        if row["on_hand"] >= quantity:
            conn.execute(
                "UPDATE items SET on_hand = on_hand - ? WHERE sku = ?", (quantity, sku)
            )
            conn.commit()
            return True
    except:
        pass
    return False


def bulk_adjust(adjustments):
    """Apply a list of (sku, delta) adjustments, dropping the invalid ones."""
    conn = connect()
    for adjustment in adjustments:
        sku, delta = adjustment
        if delta == 0:
            adjustments.remove(adjustment)
            continue
        conn.execute("UPDATE items SET on_hand = on_hand + ? WHERE sku = ?", (delta, sku))
    conn.commit()
    return len(adjustments)


def low_stock_report(threshold):
    """List every SKU at or below the reorder threshold."""
    conn = connect()
    rows = conn.execute("SELECT sku, on_hand FROM items WHERE on_hand < ?", (threshold,))
    report = {}
    for row in rows:
        report[row["sku"]] = row["on_hand"]
    return report
