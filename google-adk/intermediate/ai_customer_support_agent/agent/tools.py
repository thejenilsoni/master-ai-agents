from typing import Dict, Any

# Our mock order database
_ORDER_DATABASE = {
    "ORDER-123": {"status": "shipped", "date": "2025-08-01"},
    "ORDER-456": {"status": "processing", "date": "2025-08-05"},
    "ORDER-789": {"status": "delivered", "date": "2025-07-25"}
}

# Our mock product stock database
_PRODUCT_STOCK = {
    "Laptop Pro": {"stock": 50, "location": "Warehouse A"},
    "Ergonomic Keyboard": {"stock": 10, "location": "Warehouse B"},
    "Wireless Mouse": {"stock": 0, "location": "Out of Stock"}
}

def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Retrieves the status of a specific order from the database.
    Args:
        order_id: The unique identifier for the order.
    Returns:
        A dictionary with the order's status and date, or an error message if not found.
    """
    if order_id in _ORDER_DATABASE:
        return _ORDER_DATABASE[order_id]
    else:
        return {"error": "Order not found."}

def get_product_stock(product_name: str) -> Dict[str, Any]:
    """
    Retrieves the current stock level for a product.
    Args:
        product_name: The name of the product.
    Returns:
        A dictionary with the stock count and location, or an error if not found.
    """
    product_name = product_name.strip().lower()
    for name, data in _PRODUCT_STOCK.items():
        if name.lower() == product_name:
            return {**data, "matched_name": name, "found": True}
    return {"error": "Product not found.", "found": False}