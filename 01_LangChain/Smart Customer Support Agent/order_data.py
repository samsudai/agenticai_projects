from pathlib import Path
import pandas as pd


# Path to your CSV file
DATA_FILE = Path("data/orders.csv")


def clean_column_name(column_name):
    """
    Converts column names into a standard format.
    Example:
    'Order ID' -> 'order_id'
    'Customer Name' -> 'customer_name'
    """
    return (
        column_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_order_details(order_id):
    """
    Reads order details from data/orders.csv using the order_id.
    """

    if not order_id:
        return None

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "orders.csv not found. Please check that the file exists at data/orders.csv"
        )

    # Read CSV
    orders_df = pd.read_csv(DATA_FILE)

    # Clean column names
    orders_df.columns = [clean_column_name(col) for col in orders_df.columns]

    # Check required column
    if "order_id" not in orders_df.columns:
        raise KeyError(
            f"'order_id' column not found. Available columns are: {list(orders_df.columns)}"
        )

    # Clean order ID values
    orders_df["order_id"] = orders_df["order_id"].astype(str).str.strip().str.upper()

    search_order_id = order_id.strip().upper()

    matching_order = orders_df[orders_df["order_id"] == search_order_id]

    if matching_order.empty:
        return None

    order = matching_order.iloc[0].to_dict()

    return order