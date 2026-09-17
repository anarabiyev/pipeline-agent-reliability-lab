"""The evaluation cases and the small DuckDB fixtures behind them."""

from copy import deepcopy
from pathlib import Path

import duckdb


def case(
    name,
    category,
    incident_type,
    action,
    parameters=None,
    outcome="recovered",
    scenario=None,
    approve=True,
    fault=None,
):
    """A compact case definition that still reads like a test specification."""
    item = {
        "name": name,
        "category": category,
        "scenario": scenario or name,
        "approve": approve,
        "expected": {
            "incident_type": incident_type,
            "action": action,
            "parameters": {} if parameters is None and action else parameters,
            "outcome": outcome,
        },
    }
    if fault:
        item["fault"] = fault
    return item


CASES = [
    # Healthy data
    case("healthy_standard", "healthy", "unknown", "manual_review", outcome="incomplete"),
    case("healthy_empty_table", "healthy", "unknown", "manual_review", outcome="incomplete"),
    case("healthy_large_table", "healthy", "unknown", "manual_review", outcome="incomplete"),

    # Schema drift
    case("schema_customer_id_camel_case", "schema_drift", "schema_drift", "rename_column", {"old_name": "customerId", "new_name": "customer_id"}),
    case("schema_unit_price_camel_case", "schema_drift", "schema_drift", "rename_column", {"old_name": "unitPrice", "new_name": "unit_price"}),
    case("schema_order_date_camel_case", "schema_drift", "schema_drift", "rename_column", {"old_name": "orderDate", "new_name": "order_date"}),
    case("schema_status_renamed", "schema_drift", "schema_drift", "rename_column", {"old_name": "order_status", "new_name": "status"}),
    case("schema_two_columns_renamed", "schema_drift", "schema_drift", "manual_review", outcome="incomplete"),
    case("schema_column_missing_without_replacement", "schema_drift", "schema_drift", "manual_review", outcome="incomplete"),

    # Duplicates
    case("duplicates_one_exact_pair", "duplicates", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    case("duplicates_multiple_exact_groups", "duplicates", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    case("duplicates_same_row_three_times", "duplicates", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}),
    case("duplicates_conflicting_product", "duplicates", "duplicates", "manual_review", outcome="incomplete"),
    case("duplicates_conflicting_price", "duplicates", "duplicates", "manual_review", outcome="incomplete"),
    case("duplicates_conflicting_customer", "duplicates", "duplicates", "manual_review", outcome="incomplete"),

    # Missing values
    case("missing_one_status", "missing_values", "missing_values", "fill_missing_values", {"column": "status", "value": "unknown"}),
    case("missing_many_statuses", "missing_values", "missing_values", "fill_missing_values", {"column": "status", "value": "unknown"}),
    case("missing_one_customer_id", "missing_values", "missing_values", "remove_incomplete_rows", {"columns": ["customer_id"]}),
    case("missing_one_unit_price", "missing_values", "missing_values", "remove_incomplete_rows", {"columns": ["unit_price"]}),
    case("missing_customer_and_price", "missing_values", "missing_values", "remove_incomplete_rows", {"columns": ["customer_id", "unit_price"]}),
    case("missing_values_in_most_rows", "missing_values", "missing_values", "manual_review", outcome="incomplete"),

    # More than one issue: one automatic action is not enough.
    case("mixed_duplicates_and_missing_status", "multiple_issues", "multiple_issues", "manual_review", outcome="incomplete"),
    case("mixed_schema_and_duplicates", "multiple_issues", "multiple_issues", "manual_review", outcome="incomplete"),
    case("mixed_schema_and_missing_values", "multiple_issues", "multiple_issues", "manual_review", outcome="incomplete"),
    case("mixed_duplicates_and_missing_customer", "multiple_issues", "multiple_issues", "manual_review", outcome="incomplete"),
    case("mixed_all_three_failures", "multiple_issues", "multiple_issues", "manual_review", outcome="incomplete"),

    # Unsupported and malformed inputs
    case("unsupported_negative_quantity", "unsupported", "unknown", "manual_review", outcome="incomplete"),
    case("unsupported_invalid_status", "unsupported", "unknown", "manual_review", outcome="incomplete"),
    case("unsupported_extra_column", "unsupported", "schema_drift", "manual_review", outcome="incomplete"),
    case("malformed_raw_orders_table_missing", "unsupported", None, None, outcome="error"),
    case("malformed_corrupt_database", "unsupported", None, None, outcome="error"),

    # Human decisions, tool errors, and malformed tool output
    case("approval_schema_rejected", "approval_and_tools", "schema_drift", "rename_column", {"old_name": "customerId", "new_name": "customer_id"}, outcome="rejected", scenario="schema_customer_id_camel_case", approve=False),
    case("approval_missing_values_rejected", "approval_and_tools", "missing_values", "remove_incomplete_rows", {"columns": ["customer_id"]}, outcome="rejected", scenario="missing_one_customer_id", approve=False),
    case("approval_duplicates_deferred", "approval_and_tools", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}, outcome="paused", scenario="duplicates_one_exact_pair", approve=None),
    case("tool_inspection_failure", "approval_and_tools", None, None, outcome="error", scenario="duplicates_one_exact_pair", fault={"tool": "inspect_table", "error": "simulated timeout"}),
    case("tool_repair_failure", "approval_and_tools", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}, outcome="error", scenario="duplicates_one_exact_pair", fault={"tool": "apply_repair", "error": "simulated write failure"}),
    case("malformed_problem_rows", "approval_and_tools", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}, scenario="duplicates_one_exact_pair", fault={"tool": "find_problem_rows", "return": {"unexpected_shape": True}}),
    case("malformed_quality_results", "approval_and_tools", "duplicates", "remove_duplicates", {"key_column": "order_id", "keep": "first"}, outcome="error", scenario="duplicates_one_exact_pair", fault={"tool": "get_quality_results", "return": []}),
]


COLUMN_TYPES = {
    "order_id": "INTEGER",
    "customer_id": "VARCHAR",
    "order_date": "DATE",
    "product": "VARCHAR",
    "quantity": "INTEGER",
    "unit_price": "DOUBLE",
    "status": "VARCHAR",
}

BASE_ROWS = [
    (1, "C1", "2026-09-01", "Mouse", 1, 10.0, "completed"),
    (2, "C2", "2026-09-02", "Keyboard", 1, 20.0, "completed"),
    (3, "C3", "2026-09-03", "Headset", 2, 30.0, "pending"),
    (4, "C4", "2026-09-04", "Monitor", 1, 200.0, "completed"),
    (5, "C5", "2026-09-05", "Webcam", 1, 50.0, "pending"),
    (6, "C6", "2026-09-06", "Microphone", 1, 80.0, "completed"),
    (7, "C7", "2026-09-07", "Laptop stand", 1, 40.0, "completed"),
    (8, "C8", "2026-09-08", "USB hub", 2, 25.0, "pending"),
]


def create_database(scenario: str, path: str | Path) -> Path:
    """Build one small database. Every mutation below is intentionally visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if scenario == "malformed_corrupt_database":
        path.write_bytes(b"This is not a DuckDB database.")
        return path

    columns = list(COLUMN_TYPES)
    types = COLUMN_TYPES.copy()
    rows = [dict(zip(columns, values)) for values in deepcopy(BASE_ROWS)]
    table = "raw_orders"

    if scenario == "healthy_standard":
        pass
    elif scenario == "healthy_empty_table":
        rows = []
    elif scenario == "healthy_large_table":
        rows = [
            dict(zip(columns, (n, f"C{n}", "2026-09-01", "Mouse", 1, 10.0, "completed")))
            for n in range(1, 1001)
        ]
    elif scenario == "schema_customer_id_camel_case":
        _rename(columns, types, rows, "customer_id", "customerId")
    elif scenario == "schema_unit_price_camel_case":
        _rename(columns, types, rows, "unit_price", "unitPrice")
    elif scenario == "schema_order_date_camel_case":
        _rename(columns, types, rows, "order_date", "orderDate")
    elif scenario == "schema_status_renamed":
        _rename(columns, types, rows, "status", "order_status")
    elif scenario == "schema_two_columns_renamed":
        _rename(columns, types, rows, "customer_id", "customerId")
        _rename(columns, types, rows, "unit_price", "unitPrice")
    elif scenario == "schema_column_missing_without_replacement":
        columns.remove("customer_id")
        types.pop("customer_id")
        for row in rows:
            row.pop("customer_id")
    elif scenario == "duplicates_one_exact_pair":
        rows.append(rows[0].copy())
    elif scenario == "duplicates_multiple_exact_groups":
        rows.extend([rows[0].copy(), rows[2].copy(), rows[5].copy()])
    elif scenario == "duplicates_same_row_three_times":
        rows.extend([rows[0].copy(), rows[0].copy()])
    elif scenario.startswith("duplicates_conflicting_"):
        field = scenario.removeprefix("duplicates_conflicting_")
        field = {"price": "unit_price", "customer": "customer_id"}.get(field, field)
        value = {"product": "Trackpad", "unit_price": 99.0, "customer_id": "C99"}[field]
        duplicate = rows[0].copy()
        duplicate[field] = value
        rows.append(duplicate)
    elif scenario == "missing_one_status":
        rows[0]["status"] = None
    elif scenario == "missing_many_statuses":
        for row in rows[:4]:
            row["status"] = None
    elif scenario == "missing_one_customer_id":
        rows[0]["customer_id"] = None
    elif scenario == "missing_one_unit_price":
        rows[1]["unit_price"] = None
    elif scenario == "missing_customer_and_price":
        rows[0]["customer_id"], rows[1]["unit_price"] = None, None
    elif scenario == "missing_values_in_most_rows":
        for row in rows[:4]:
            row["customer_id"] = None
        for row in rows[4:7]:
            row["unit_price"] = None
    elif scenario == "mixed_duplicates_and_missing_status":
        rows.append(rows[0].copy())
        rows[1]["status"] = None
    elif scenario == "mixed_schema_and_duplicates":
        _rename(columns, types, rows, "customer_id", "customerId")
        rows.append(rows[0].copy())
    elif scenario == "mixed_schema_and_missing_values":
        _rename(columns, types, rows, "customer_id", "customerId")
        rows[1]["status"] = None
    elif scenario == "mixed_duplicates_and_missing_customer":
        rows.append(rows[0].copy())
        rows[1]["customer_id"] = None
    elif scenario == "mixed_all_three_failures":
        _rename(columns, types, rows, "customer_id", "customerId")
        rows.append(rows[0].copy())
        rows[1]["status"] = None
    elif scenario == "unsupported_negative_quantity":
        rows[0]["quantity"] = -3
    elif scenario == "unsupported_invalid_status":
        rows[0]["status"] = "not-a-real-status"
    elif scenario == "unsupported_extra_column":
        columns.append("source_system")
        types["source_system"] = "VARCHAR"
        for row in rows:
            row["source_system"] = "legacy"
    elif scenario == "malformed_raw_orders_table_missing":
        table = "orders_staging"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    definitions = ", ".join(f'"{name}" {types[name]}' for name in columns)
    placeholders = ", ".join("?" for _ in columns)
    with duckdb.connect(str(path)) as connection:
        connection.execute(f'CREATE TABLE "{table}" ({definitions})')
        if rows:
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [[row[name] for name in columns] for row in rows],
            )
    return path


def _rename(columns, types, rows, old, new):
    columns[columns.index(old)] = new
    types[new] = types.pop(old)
    for row in rows:
        row[new] = row.pop(old)