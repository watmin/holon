#!/usr/bin/env python3
"""
Realistic Data Generator for Challenge 010

Creates messy, production-like data with:
- Multiple intermingling schemas (5-10 types)
- Missing fields (10-30% optional field omission)
- Extra unexpected fields (5-10%)
- Deep nesting (2-4 levels)
- Variable-length lists (0-20 items)
- Type variance (null vs missing, int vs string)
- High cardinality values
- Temporal patterns
- Correlated noise (not just random)

Goal: Stress-test VSA/HDC with data that actually looks real.
"""

import hashlib
import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SchemaConfig:
    """Configuration for a schema type."""
    name: str
    required_fields: Dict[str, str]  # field_name -> type
    optional_fields: Dict[str, str]  # field_name -> type
    nested_schemas: Dict[str, str] = field(default_factory=dict)  # field_name -> schema_name
    list_fields: Dict[str, str] = field(default_factory=dict)  # field_name -> item_type
    category_field: Optional[str] = None  # Field that determines category
    category_values: List[str] = field(default_factory=list)


class RealisticDataGenerator:
    """
    Generates realistic, messy data for VSA/HDC stress testing.

    Key features:
    - Deterministic: same seed = same data (for distributed consensus testing)
    - High cardinality: many unique values per field
    - Structural variance: missing fields, extra fields, type coercion
    - Temporal patterns: realistic timestamps with patterns
    - Correlated noise: noise that has structure (not pure random)
    """

    # Schema definitions - these intermingle in the generated data
    SCHEMAS = {
        "api_request": SchemaConfig(
            name="api_request",
            required_fields={
                "method": "http_method",
                "path": "api_path",
                "status_code": "http_status",
                "timestamp": "timestamp",
            },
            optional_fields={
                "user_id": "user_id",
                "session_id": "session_id",
                "latency_ms": "latency",
                "request_id": "uuid",
                "user_agent": "user_agent",
                "ip_address": "ip_address",
                "error_message": "error_text",
            },
            nested_schemas={
                "headers": "http_headers",
                "query_params": "key_value_pairs",
            },
            category_field="path",
            category_values=[
                "/api/users", "/api/orders", "/api/products",
                "/api/auth/login", "/api/auth/logout", "/api/payments",
                "/api/search", "/api/notifications", "/api/settings",
                "/api/reports", "/api/analytics", "/api/webhooks",
            ],
        ),
        "log_entry": SchemaConfig(
            name="log_entry",
            required_fields={
                "level": "log_level",
                "message": "log_message",
                "timestamp": "timestamp",
                "service": "service_name",
            },
            optional_fields={
                "trace_id": "trace_id",
                "span_id": "span_id",
                "error_code": "error_code",
                "stack_trace": "stack_trace",
                "duration_ms": "latency",
                "host": "hostname",
                "pid": "pid",
                "thread_id": "thread_id",
            },
            nested_schemas={
                "context": "key_value_pairs",
                "tags": "tag_list",
            },
            category_field="level",
            category_values=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        ),
        "user_event": SchemaConfig(
            name="user_event",
            required_fields={
                "event_type": "event_type",
                "user_id": "user_id",
                "timestamp": "timestamp",
            },
            optional_fields={
                "session_id": "session_id",
                "device_type": "device_type",
                "os": "os_name",
                "browser": "browser_name",
                "country": "country_code",
                "city": "city_name",
                "referrer": "url",
                "campaign_id": "campaign_id",
            },
            nested_schemas={
                "properties": "event_properties",
                "user_traits": "user_traits",
            },
            list_fields={
                "previous_events": "event_type",
            },
            category_field="event_type",
            category_values=[
                "page_view", "click", "scroll", "form_submit", "purchase",
                "signup", "login", "logout", "search", "add_to_cart",
                "remove_from_cart", "checkout_start", "checkout_complete",
                "error", "video_play", "video_pause", "share", "download",
            ],
        ),
        "order": SchemaConfig(
            name="order",
            required_fields={
                "order_id": "order_id",
                "customer_id": "user_id",
                "status": "order_status",
                "total": "currency_amount",
                "created_at": "timestamp",
            },
            optional_fields={
                "updated_at": "timestamp",
                "shipped_at": "timestamp",
                "delivered_at": "timestamp",
                "cancelled_at": "timestamp",
                "discount_code": "discount_code",
                "discount_amount": "currency_amount",
                "notes": "free_text",
                "priority": "priority_level",
            },
            nested_schemas={
                "shipping_address": "address",
                "billing_address": "address",
                "payment": "payment_info",
            },
            list_fields={
                "items": "order_item",
            },
            category_field="status",
            category_values=[
                "pending", "confirmed", "processing", "shipped",
                "delivered", "cancelled", "refunded", "on_hold",
            ],
        ),
        "metric": SchemaConfig(
            name="metric",
            required_fields={
                "name": "metric_name",
                "value": "metric_value",
                "timestamp": "timestamp",
            },
            optional_fields={
                "unit": "metric_unit",
                "host": "hostname",
                "service": "service_name",
                "environment": "environment",
                "region": "region",
            },
            nested_schemas={
                "tags": "metric_tags",
                "metadata": "key_value_pairs",
            },
            category_field="name",
            category_values=[
                "cpu_usage", "memory_usage", "disk_io", "network_in",
                "network_out", "request_count", "error_rate", "latency_p50",
                "latency_p99", "queue_depth", "connection_count", "cache_hit_rate",
            ],
        ),
        "alert": SchemaConfig(
            name="alert",
            required_fields={
                "alert_id": "uuid",
                "severity": "severity",
                "title": "alert_title",
                "triggered_at": "timestamp",
            },
            optional_fields={
                "resolved_at": "timestamp",
                "acknowledged_by": "user_id",
                "acknowledged_at": "timestamp",
                "runbook_url": "url",
                "dashboard_url": "url",
                "notes": "free_text",
            },
            nested_schemas={
                "source": "alert_source",
                "conditions": "alert_conditions",
            },
            list_fields={
                "affected_services": "service_name",
                "related_alerts": "uuid",
            },
            category_field="severity",
            category_values=["info", "warning", "critical", "emergency"],
        ),
        "config_change": SchemaConfig(
            name="config_change",
            required_fields={
                "change_id": "uuid",
                "key": "config_key",
                "timestamp": "timestamp",
                "changed_by": "user_id",
            },
            optional_fields={
                "old_value": "config_value",
                "new_value": "config_value",
                "reason": "free_text",
                "approved_by": "user_id",
                "rollback_of": "uuid",
            },
            nested_schemas={
                "metadata": "key_value_pairs",
            },
            category_field="key",
            category_values=[
                "feature_flags.new_checkout", "feature_flags.dark_mode",
                "rate_limits.api", "rate_limits.auth",
                "cache.ttl", "cache.size",
                "database.pool_size", "database.timeout",
                "logging.level", "logging.sampling_rate",
            ],
        ),
        "deployment": SchemaConfig(
            name="deployment",
            required_fields={
                "deployment_id": "uuid",
                "service": "service_name",
                "version": "version",
                "environment": "environment",
                "started_at": "timestamp",
            },
            optional_fields={
                "completed_at": "timestamp",
                "status": "deployment_status",
                "deployed_by": "user_id",
                "commit_sha": "commit_sha",
                "rollback": "boolean",
                "canary_percentage": "percentage",
            },
            nested_schemas={
                "health_checks": "health_check_results",
                "metrics_before": "deployment_metrics",
                "metrics_after": "deployment_metrics",
            },
            list_fields={
                "changed_files": "file_path",
                "affected_hosts": "hostname",
            },
            category_field="status",
            category_values=[
                "pending", "in_progress", "succeeded", "failed",
                "rolled_back", "cancelled",
            ],
        ),
    }

    # Nested schema definitions
    NESTED_SCHEMAS = {
        "http_headers": {
            "Content-Type": "content_type",
            "Accept": "content_type",
            "Authorization": "auth_header",
            "X-Request-ID": "uuid",
            "X-Correlation-ID": "trace_id",
        },
        "key_value_pairs": "dynamic",  # Variable keys
        "address": {
            "street": "street_address",
            "city": "city_name",
            "state": "state_code",
            "zip": "zip_code",
            "country": "country_code",
        },
        "payment_info": {
            "method": "payment_method",
            "last_four": "card_last_four",
            "processor": "payment_processor",
        },
        "order_item": {
            "product_id": "product_id",
            "name": "product_name",
            "quantity": "quantity",
            "price": "currency_amount",
        },
        "event_properties": "dynamic",
        "user_traits": {
            "plan": "plan_type",
            "signup_date": "date",
            "lifetime_value": "currency_amount",
        },
        "tag_list": "list_of_tags",
        "metric_tags": "dynamic",
        "alert_source": {
            "type": "alert_source_type",
            "name": "metric_name",
            "threshold": "metric_value",
            "current_value": "metric_value",
        },
        "alert_conditions": "dynamic",
        "health_check_results": {
            "status": "health_status",
            "latency_ms": "latency",
            "checks_passed": "quantity",
            "checks_failed": "quantity",
        },
        "deployment_metrics": {
            "error_rate": "percentage",
            "latency_p50": "latency",
            "latency_p99": "latency",
            "throughput": "metric_value",
        },
    }

    def __init__(
        self,
        seed: int = 42,
        cardinality: int = 10000,
        missing_field_rate: float = 0.15,
        extra_field_rate: float = 0.08,
        type_coercion_rate: float = 0.05,
        null_vs_missing_rate: float = 0.5,  # When optional is missing, 50% null vs 50% absent
        max_list_length: int = 15,
        max_nesting_depth: int = 3,
    ):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility (same seed = same data)
            cardinality: Number of unique values per high-cardinality field
            missing_field_rate: Probability of omitting optional fields
            extra_field_rate: Probability of adding unexpected fields
            type_coercion_rate: Probability of type variance (int as string, etc.)
            null_vs_missing_rate: When optional missing, chance of explicit null vs absent
            max_list_length: Maximum items in list fields
            max_nesting_depth: Maximum nesting depth for recursive structures
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.cardinality = cardinality
        self.missing_field_rate = missing_field_rate
        self.extra_field_rate = extra_field_rate
        self.type_coercion_rate = type_coercion_rate
        self.null_vs_missing_rate = null_vs_missing_rate
        self.max_list_length = max_list_length
        self.max_nesting_depth = max_nesting_depth

        # Pre-generate high-cardinality value pools
        self._value_pools = self._build_value_pools()

        # Base timestamp for temporal patterns
        self._base_time = datetime(2026, 1, 1, 0, 0, 0)

        # Track unique atoms generated (for cardinality reporting)
        self.unique_atoms = set()

    def _build_value_pools(self) -> Dict[str, List[str]]:
        """Pre-generate pools of values for high-cardinality fields."""
        pools = {}

        # User IDs - UUIDs with some repetition (power law)
        pools["user_id"] = [f"usr_{i:08x}" for i in range(self.cardinality)]

        # Session IDs
        pools["session_id"] = [f"sess_{i:012x}" for i in range(self.cardinality)]

        # Product IDs
        pools["product_id"] = [f"prod_{i:06d}" for i in range(self.cardinality // 10)]

        # Order IDs
        pools["order_id"] = [f"ord_{i:010d}" for i in range(self.cardinality)]

        # API paths with parameters
        base_paths = ["/api/users", "/api/orders", "/api/products", "/api/search"]
        pools["api_path"] = base_paths + [
            f"{p}/{i}" for p in base_paths for i in range(100)
        ]

        # Service names
        pools["service_name"] = [
            "auth-service", "user-service", "order-service", "payment-service",
            "notification-service", "search-service", "analytics-service",
            "gateway", "worker-1", "worker-2", "worker-3", "scheduler",
            "cache-service", "db-proxy", "load-balancer", "cdn-origin",
        ]

        # Hostnames
        pools["hostname"] = [
            f"host-{region}-{i:03d}"
            for region in ["us-east", "us-west", "eu-west", "ap-south"]
            for i in range(50)
        ]

        # IP addresses
        pools["ip_address"] = [
            f"10.{a}.{b}.{c}"
            for a in range(1, 10)
            for b in range(256)
            for c in range(1, 255)
        ][:self.cardinality]

        # User agents
        browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        versions = ["100", "101", "102", "103", "104", "105"]
        pools["user_agent"] = [
            f"Mozilla/5.0 ({os}) {b}/{v}"
            for os in ["Windows NT 10.0", "Macintosh", "X11; Linux x86_64"]
            for b in browsers
            for v in versions
        ]

        # Trace IDs
        pools["trace_id"] = [f"{i:032x}" for i in range(self.cardinality)]
        pools["span_id"] = [f"{i:016x}" for i in range(self.cardinality)]

        # Error messages with variation
        error_templates = [
            "Connection refused to {0}",
            "Timeout after 30000ms waiting for {0}",
            "Invalid {0} in request",
            "Permission denied for user {0}",
            "Resource {0} not found",
            "Rate limit exceeded for {0}",
            "Database error: {0}",
            "Validation failed: {0}",
        ]
        pools["error_text"] = [
            t.format(f"resource_{i}")
            for t in error_templates
            for i in range(100)
        ]

        # Log messages
        log_templates = [
            "Processing request {}",
            "User {} logged in",
            "Order {} created",
            "Cache miss for key {}",
            "Retrying operation, attempt {}",
            "Connection pool exhausted, waiting",
            "Background job {} started",
            "Metrics flushed: {} points",
        ]
        pools["log_message"] = [
            t.format(i) if "{}" in t else t
            for t in log_templates
            for i in range(100)
        ]

        # City names
        pools["city_name"] = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "San Francisco", "Seattle", "Boston", "Denver", "Austin",
            "Portland", "Miami", "Atlanta", "Detroit", "Minneapolis",
            "London", "Paris", "Berlin", "Tokyo", "Sydney",
            "Toronto", "Vancouver", "Mumbai", "Singapore", "Dubai",
        ]

        # Config keys
        pools["config_key"] = [
            f"{ns}.{key}"
            for ns in ["feature_flags", "rate_limits", "cache", "database", "logging", "security"]
            for key in ["enabled", "threshold", "timeout", "size", "level", "mode"]
        ]

        # Versions
        pools["version"] = [f"v{major}.{minor}.{patch}"
                           for major in range(1, 5)
                           for minor in range(20)
                           for patch in range(10)]

        # Commit SHAs
        pools["commit_sha"] = [f"{i:040x}" for i in range(self.cardinality)]

        # File paths
        pools["file_path"] = [
            f"src/{module}/{file}.py"
            for module in ["auth", "api", "models", "utils", "services", "handlers"]
            for file in ["main", "helpers", "config", "tests", "types"]
        ]

        return pools

    def _get_value(self, value_type: str, record_seed: int) -> Any:
        """Generate a value of the specified type."""
        # Use record_seed for per-record determinism
        local_rng = random.Random(record_seed ^ hash(value_type))

        # Check if we have a pre-generated pool
        if value_type in self._value_pools:
            # Power-law distribution: some values more common than others
            pool = self._value_pools[value_type]
            # Zipf-like: pick from smaller index more often
            idx = min(int(local_rng.paretovariate(1.5)), len(pool) - 1)
            value = pool[idx]
            self.unique_atoms.add(value)
            return value

        # Generate based on type
        if value_type == "timestamp":
            # Temporal pattern: clustered around certain times
            offset_days = local_rng.gauss(30, 15)  # Clustered around 30 days ago
            offset_hours = local_rng.gauss(12, 4)  # Clustered around noon
            offset_minutes = local_rng.random() * 60
            ts = self._base_time + timedelta(
                days=offset_days, hours=offset_hours, minutes=offset_minutes
            )
            return ts.isoformat() + "Z"

        elif value_type == "http_method":
            # Realistic distribution
            return local_rng.choices(
                ["GET", "POST", "PUT", "DELETE", "PATCH"],
                weights=[60, 25, 8, 5, 2]
            )[0]

        elif value_type == "http_status":
            return local_rng.choices(
                [200, 201, 204, 301, 400, 401, 403, 404, 500, 502, 503],
                weights=[70, 5, 3, 2, 5, 3, 2, 5, 2, 2, 1]
            )[0]

        elif value_type == "log_level":
            return local_rng.choices(
                ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
                weights=[5, 70, 15, 8, 2]
            )[0]

        elif value_type == "latency":
            # Log-normal distribution (most fast, some slow)
            return max(1, int(local_rng.lognormvariate(3, 1.5)))

        elif value_type == "currency_amount":
            # Price distribution
            return round(local_rng.lognormvariate(3, 1) * 10, 2)

        elif value_type == "metric_value":
            return round(local_rng.gauss(50, 20), 2)

        elif value_type == "percentage":
            return round(local_rng.random() * 100, 1)

        elif value_type == "quantity":
            return local_rng.randint(1, 20)

        elif value_type == "uuid":
            # Deterministic UUID from seed
            return str(uuid.UUID(int=local_rng.getrandbits(128)))

        elif value_type == "boolean":
            return local_rng.choice([True, False])

        elif value_type in ["priority_level", "severity"]:
            return local_rng.choice(["low", "medium", "high", "critical"])

        elif value_type == "environment":
            return local_rng.choices(
                ["production", "staging", "development"],
                weights=[70, 20, 10]
            )[0]

        elif value_type == "region":
            return local_rng.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"])

        elif value_type == "free_text":
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                    "error", "warning", "info", "debug", "processing", "completed",
                    "failed", "success", "retry", "timeout", "connection", "request"]
            n_words = local_rng.randint(3, 15)
            return " ".join(local_rng.choices(words, k=n_words))

        elif value_type == "date":
            days_ago = local_rng.randint(0, 365)
            dt = self._base_time - timedelta(days=days_ago)
            return dt.strftime("%Y-%m-%d")

        elif value_type == "content_type":
            return local_rng.choice([
                "application/json", "text/html", "text/plain",
                "application/xml", "multipart/form-data"
            ])

        elif value_type == "auth_header":
            return f"Bearer {local_rng.getrandbits(128):032x}"

        elif value_type == "payment_method":
            return local_rng.choice(["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"])

        elif value_type == "card_last_four":
            return f"{local_rng.randint(0, 9999):04d}"

        elif value_type == "payment_processor":
            return local_rng.choice(["stripe", "braintree", "adyen", "square"])

        elif value_type == "plan_type":
            return local_rng.choice(["free", "starter", "pro", "enterprise"])

        elif value_type == "event_type":
            return local_rng.choice([
                "page_view", "click", "scroll", "form_submit", "purchase",
                "signup", "login", "logout", "search", "add_to_cart"
            ])

        elif value_type == "device_type":
            return local_rng.choices(
                ["desktop", "mobile", "tablet"],
                weights=[50, 40, 10]
            )[0]

        elif value_type == "os_name":
            return local_rng.choice(["Windows", "macOS", "Linux", "iOS", "Android"])

        elif value_type == "browser_name":
            return local_rng.choice(["Chrome", "Firefox", "Safari", "Edge"])

        elif value_type == "country_code":
            return local_rng.choice(["US", "GB", "DE", "FR", "JP", "IN", "BR", "CA", "AU"])

        elif value_type == "state_code":
            return local_rng.choice(["CA", "NY", "TX", "FL", "WA", "OR", "CO", "MA"])

        elif value_type == "zip_code":
            return f"{local_rng.randint(10000, 99999)}"

        elif value_type == "street_address":
            return f"{local_rng.randint(1, 9999)} {local_rng.choice(['Main', 'Oak', 'Pine', 'Elm', 'First'])} St"

        elif value_type == "url":
            domains = ["example.com", "internal.net", "docs.company.io"]
            paths = ["dashboard", "docs", "help", "runbook", "metrics"]
            return f"https://{local_rng.choice(domains)}/{local_rng.choice(paths)}"

        elif value_type == "campaign_id":
            return f"camp_{local_rng.randint(1000, 9999)}"

        elif value_type == "discount_code":
            return f"{''.join(local_rng.choices(string.ascii_uppercase, k=6))}{local_rng.randint(10, 99)}"

        elif value_type == "error_code":
            return f"ERR_{local_rng.randint(1000, 9999)}"

        elif value_type == "stack_trace":
            return f"at Function.{local_rng.choice(['handle', 'process', 'execute'])}(...)"

        elif value_type == "thread_id":
            return local_rng.randint(1, 100)

        elif value_type == "pid":
            return local_rng.randint(1000, 65535)

        elif value_type == "metric_name":
            return local_rng.choice([
                "cpu_usage", "memory_usage", "disk_io", "network_in",
                "network_out", "request_count", "error_rate", "latency_p99"
            ])

        elif value_type == "metric_unit":
            return local_rng.choice(["percent", "bytes", "ms", "count", "req/s"])

        elif value_type == "alert_title":
            templates = [
                "High {} on {}", "Low {} threshold", "{} exceeded limit",
                "{} degraded", "Anomaly detected in {}"
            ]
            return local_rng.choice(templates).format(
                local_rng.choice(["latency", "error rate", "CPU", "memory"]),
                local_rng.choice(["prod", "staging", "api-gateway"])
            )

        elif value_type == "alert_source_type":
            return local_rng.choice(["metric", "log", "synthetic", "trace"])

        elif value_type == "health_status":
            return local_rng.choices(["healthy", "degraded", "unhealthy"], weights=[80, 15, 5])[0]

        elif value_type == "deployment_status":
            return local_rng.choice(["pending", "in_progress", "succeeded", "failed", "rolled_back"])

        elif value_type == "order_status":
            return local_rng.choice(["pending", "confirmed", "shipped", "delivered", "cancelled"])

        elif value_type == "config_value":
            # Mix of types
            choice = local_rng.randint(0, 3)
            if choice == 0:
                return local_rng.choice([True, False])
            elif choice == 1:
                return local_rng.randint(1, 1000)
            elif choice == 2:
                return round(local_rng.random(), 3)
            else:
                return local_rng.choice(["enabled", "disabled", "auto"])

        elif value_type == "product_name":
            adjectives = ["Premium", "Basic", "Pro", "Ultra", "Lite"]
            nouns = ["Widget", "Gadget", "Tool", "Service", "Package"]
            return f"{local_rng.choice(adjectives)} {local_rng.choice(nouns)}"

        else:
            # Fallback: generate a generic value
            value = f"{value_type}_{local_rng.randint(0, self.cardinality)}"
            self.unique_atoms.add(value)
            return value

    def _generate_nested(
        self, schema_name: str, record_seed: int, depth: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Generate a nested object based on schema."""
        if depth >= self.max_nesting_depth:
            return None

        local_rng = random.Random(record_seed ^ hash(schema_name) ^ depth)

        if schema_name not in self.NESTED_SCHEMAS:
            return None

        schema = self.NESTED_SCHEMAS[schema_name]

        if schema == "dynamic":
            # Dynamic key-value pairs
            n_pairs = local_rng.randint(1, 5)
            result = {}
            for i in range(n_pairs):
                key = f"key_{local_rng.randint(1, 100)}"
                # Random value type
                val_type = local_rng.choice(["string", "int", "float", "bool"])
                if val_type == "string":
                    result[key] = f"value_{local_rng.randint(1, 1000)}"
                elif val_type == "int":
                    result[key] = local_rng.randint(1, 1000)
                elif val_type == "float":
                    result[key] = round(local_rng.random() * 100, 2)
                else:
                    result[key] = local_rng.choice([True, False])
            return result

        elif schema == "list_of_tags":
            n_tags = local_rng.randint(1, 8)
            return [f"tag_{local_rng.randint(1, 50)}" for _ in range(n_tags)]

        elif isinstance(schema, dict):
            result = {}
            for field_name, field_type in schema.items():
                # Sometimes skip optional nested fields
                if local_rng.random() < self.missing_field_rate:
                    continue
                result[field_name] = self._get_value(field_type, record_seed + hash(field_name))
            return result if result else None

        return None

    def _generate_list_field(
        self, item_type: str, record_seed: int
    ) -> List[Any]:
        """Generate a list of items."""
        local_rng = random.Random(record_seed ^ hash(item_type))

        # Variable length with power-law (most short, some long)
        n_items = min(
            int(local_rng.paretovariate(2)) + 1,
            self.max_list_length
        )

        items = []
        for i in range(n_items):
            item_seed = record_seed + i * 1000
            if item_type in self.NESTED_SCHEMAS:
                item = self._generate_nested(item_type, item_seed)
                if item:
                    items.append(item)
            else:
                items.append(self._get_value(item_type, item_seed))

        return items

    def _maybe_coerce_type(self, value: Any, local_rng: random.Random) -> Any:
        """Sometimes coerce types for realistic messiness."""
        if local_rng.random() > self.type_coercion_rate:
            return value

        if isinstance(value, int):
            # Int as string
            return str(value)
        elif isinstance(value, float):
            # Float as string or truncated int
            return str(value) if local_rng.random() > 0.5 else int(value)
        elif isinstance(value, bool):
            # Bool as string
            return "true" if value else "false"

        return value

    def generate_record(
        self,
        schema_name: str = None,
        record_id: int = 0
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Generate a single record.

        Args:
            schema_name: Schema to use (None = random)
            record_id: Unique ID for deterministic generation

        Returns:
            Tuple of (record, schema_name, category)
        """
        record_seed = self.seed + record_id * 7919  # Prime multiplier for spread
        local_rng = random.Random(record_seed)

        # Pick schema
        if schema_name is None:
            schema_name = local_rng.choice(list(self.SCHEMAS.keys()))

        schema = self.SCHEMAS[schema_name]
        record = {}

        # Add schema type as a field (for classification)
        record["_schema"] = schema_name

        # Required fields
        for field_name, field_type in schema.required_fields.items():
            value = self._get_value(field_type, record_seed + hash(field_name))
            record[field_name] = self._maybe_coerce_type(value, local_rng)

        # Optional fields (sometimes missing)
        for field_name, field_type in schema.optional_fields.items():
            if local_rng.random() < self.missing_field_rate:
                # Missing - sometimes null, sometimes absent
                if local_rng.random() < self.null_vs_missing_rate:
                    record[field_name] = None
                # else: field is simply absent
                continue
            value = self._get_value(field_type, record_seed + hash(field_name))
            record[field_name] = self._maybe_coerce_type(value, local_rng)

        # Nested schemas
        for field_name, nested_schema in schema.nested_schemas.items():
            if local_rng.random() < self.missing_field_rate:
                continue
            nested = self._generate_nested(nested_schema, record_seed + hash(field_name))
            if nested:
                record[field_name] = nested

        # List fields
        for field_name, item_type in schema.list_fields.items():
            if local_rng.random() < self.missing_field_rate:
                # Empty list vs missing
                if local_rng.random() < 0.5:
                    record[field_name] = []
                continue
            record[field_name] = self._generate_list_field(
                item_type, record_seed + hash(field_name)
            )

        # Extra unexpected fields (chaos)
        if local_rng.random() < self.extra_field_rate:
            n_extra = local_rng.randint(1, 3)
            for i in range(n_extra):
                extra_key = f"_extra_{local_rng.choice(['debug', 'internal', 'meta', 'temp'])}_{i}"
                extra_val = local_rng.choice([
                    local_rng.randint(1, 1000),
                    f"extra_value_{local_rng.randint(1, 100)}",
                    local_rng.random() > 0.5,
                    None,
                ])
                record[extra_key] = extra_val

        # Determine category from category field
        category = schema_name  # Default: schema is category
        if schema.category_field and schema.category_field in record:
            category = f"{schema_name}:{record[schema.category_field]}"

        return record, schema_name, category

    def generate_dataset(
        self,
        n_samples: int,
        schema_distribution: Dict[str, float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """
        Generate a complete dataset.

        Args:
            n_samples: Number of records to generate
            schema_distribution: Optional weights for schema selection

        Returns:
            Tuple of (records, schema_names, categories)
        """
        records = []
        schemas = []
        categories = []

        schema_names = list(self.SCHEMAS.keys())
        if schema_distribution:
            weights = [schema_distribution.get(s, 1.0) for s in schema_names]
        else:
            weights = [1.0] * len(schema_names)

        for i in range(n_samples):
            local_rng = random.Random(self.seed + i)
            schema_name = local_rng.choices(schema_names, weights=weights)[0]

            record, schema, category = self.generate_record(schema_name, i)
            records.append(record)
            schemas.append(schema)
            categories.append(category)

            if (i + 1) % 100000 == 0:
                print(f"  Generated {i+1:,} records...", flush=True)

        return records, schemas, categories

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about generated data."""
        return {
            "unique_atoms": len(self.unique_atoms),
            "schemas": list(self.SCHEMAS.keys()),
            "cardinality_target": self.cardinality,
            "missing_field_rate": self.missing_field_rate,
            "extra_field_rate": self.extra_field_rate,
        }


def demo():
    """Demo the generator."""
    print("=" * 70)
    print("Realistic Data Generator Demo")
    print("=" * 70)

    gen = RealisticDataGenerator(
        seed=42,
        cardinality=10000,
        missing_field_rate=0.20,
        extra_field_rate=0.10,
    )

    # Generate samples from each schema
    print("\nSample records from each schema:\n")
    for schema_name in gen.SCHEMAS:
        record, _, category = gen.generate_record(schema_name, record_id=hash(schema_name))
        print(f"--- {schema_name} (category: {category}) ---")
        # Pretty print, truncating long values
        for k, v in record.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            print(f"  {k}: {v_str}")
        print()

    # Generate a small dataset
    print("\nGenerating 10,000 records...")
    records, schemas, categories = gen.generate_dataset(10000)

    # Stats
    stats = gen.get_stats()
    print(f"\nStats:")
    print(f"  Total records: {len(records):,}")
    print(f"  Unique atoms observed: {stats['unique_atoms']:,}")
    print(f"  Schemas: {len(stats['schemas'])}")

    # Schema distribution
    from collections import Counter
    schema_counts = Counter(schemas)
    print(f"\nSchema distribution:")
    for s, c in schema_counts.most_common():
        print(f"  {s}: {c:,} ({100*c/len(records):.1f}%)")

    # Category distribution (top 20)
    cat_counts = Counter(categories)
    print(f"\nTop 20 categories (of {len(cat_counts)} total):")
    for cat, c in cat_counts.most_common(20):
        print(f"  {cat}: {c:,}")

    # Field presence analysis
    print("\nField presence analysis (sample of 100):")
    sample = records[:100]
    all_fields = set()
    for r in sample:
        all_fields.update(r.keys())

    field_counts = {f: sum(1 for r in sample if f in r) for f in all_fields}
    sorted_fields = sorted(field_counts.items(), key=lambda x: -x[1])
    for field, count in sorted_fields[:15]:
        print(f"  {field}: {count}% present")


if __name__ == "__main__":
    demo()
