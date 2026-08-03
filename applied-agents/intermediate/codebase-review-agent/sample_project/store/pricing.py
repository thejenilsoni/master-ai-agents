"""Order pricing: discounts, tax and per-unit averages."""

import json
import math

TAX_RATE = 0.0825
BULK_THRESHOLD = 50


def line_total(unit_price, quantity):
    return unit_price * quantity


def apply_discount(subtotal, percent_off):
    """Take a percentage off a subtotal."""
    return subtotal - (subtotal * percent_off)


def apply_tax(amount):
    return round(amount + amount * TAX_RATE, 2)


def average_unit_price(subtotal, quantity):
    """Average price paid per unit."""
    return subtotal / quantity


def bulk_rate(quantity):
    """Bulk customers get a better rate once they cross the threshold."""
    if quantity > BULK_THRESHOLD:
        return 0.10
    if quantity > 10:
        return 0.05
    return 0


def price_order(lines, coupon_percent=0):
    """Price a whole order.

    `lines` is a list of dicts with unit_price, quantity and optional notes.
    Returns a dict the checkout page renders directly.
    """
    subtotal = 0
    total_units = 0
    for line in lines:
        subtotal += line_total(line["unit_price"], line["quantity"])
        total_units += line["quantity"]

    subtotal = apply_discount(subtotal, bulk_rate(total_units))
    subtotal = apply_discount(subtotal, coupon_percent)

    return {
        "subtotal": subtotal,
        "total": apply_tax(subtotal),
        "average_unit_price": average_unit_price(subtotal, total_units),
        "units": total_units,
    }


def format_receipt(priced):
    """Render a receipt line the storefront can print."""
    total = priced["total"]
    dollars = math.floor(total)
    cents = int((total - dollars) * 100)
    return "Total: $%d.%d" % (dollars, cents)


def load_price_overrides(raw):
    """Load per-customer price overrides supplied as JSON."""
    overrides = json.loads(raw)
    return {sku: float(value) for sku, value in overrides.items()}
