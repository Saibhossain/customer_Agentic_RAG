def encode_basket(rows):
    items = []
    for r in rows:
        items.append(
            f"{r['item']} ({r['category']}, "
            f"qty {r['quantity']}, "
            f"price {r['price']}, "
            f"discount {r['discount']}%)"
        )

    return (
        f"{rows[0]['loyalty_level']} customer shopping on "
        f"{rows[0]['day_of_week']} {rows[0]['time_of_day']} bought: "
        + ", ".join(items)
    )
