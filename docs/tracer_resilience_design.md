# Design Document: Tracer Worker Resilience

**Author:** Claude
**Date:** 2025-11-10
**Status:** Proposed

## Problem Statement

The LLMTracer background worker (`rllm/sdk/tracing.py`) crashes when storage operations fail after exhausting retries (3 attempts). Once crashed, the worker thread dies and no further traces can be persisted, causing silent data loss.

### Current Behavior

1. Worker processes traces from a queue in batches
2. For each batch, calls `_store_batch_with_retry()` which retries up to 3 times
3. If all retries fail, an exception is raised
4. **Critical Issue:** The exception propagates to `_async_worker()` which has **no exception handling**
5. The unhandled exception crashes the coroutine and stops the event loop
6. The worker thread remains "alive" (thread exists) but is non-functional
7. New traces queue up until the queue fills (10,000 items), then silently drop

### Root Causes

**Location:** `rllm/sdk/tracing.py:160-201` (`_async_worker` method)

```python
async def _async_worker(self):
    """Process trace items in batches sequentially (strict FIFO persistence with batching)."""
    batch_count = 0
    while True:
        # ... batch collection logic ...

        # NO EXCEPTION HANDLING HERE!
        if len(batch) == 1:
            await self._store_trace_with_retry(batch[0])  # Can raise after 3 retries
            self._trace_queue.task_done()
        else:
            await self._store_batch_with_retry(batch)  # Can raise after 3 retries
            for _ in batch:
                self._trace_queue.task_done()
```

**Failure scenarios:**
- Context store endpoint unreachable (network issues)
- API key expired/invalid (authentication failures)
- Service overload (timeout/rate limiting)
- Malformed data (serialization errors)

## Design Goals

1. **Zero trace loss due to worker crashes** - Worker must stay alive even after storage failures
2. **Visibility** - Operators must know when traces are failing/dropping
3. **Recoverability** - Worker should recover automatically when storage becomes available
4. **Minimal performance impact** - Solution should not add significant latency
5. **Backward compatibility** - No breaking API changes to LLMTracer

## Proposed Solutions

### Option 1: Graceful Degradation with Dead Letter Queue (Recommended)

**Overview:** Wrap the worker loop in exception handling that catches storage failures and moves failed batches to a dead letter queue (DLQ) for later retry or inspection.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMTracer Worker                         │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Trace   │───▶│    Batch     │───▶│    Store to    │  │
│  │  Queue   │    │  Processing  │    │  ContextStore  │  │
│  └──────────┘    └──────────────┘    └────────────────┘  │
│                          │                     │           │
│                          │ Storage Fails       │           │
│                          ▼                     │           │
│                  ┌───────────────┐             │           │
│                  │  Dead Letter  │             │           │
│                  │     Queue     │             │           │
│                  └───────────────┘             │           │
│                          │                     │           │
│                          │ Periodic Retry      │           │
│                          └─────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Exception Wrapper in Worker Loop**
   - Wrap batch storage in try/except
   - Catch storage failures after retries exhausted
   - Log error with full context (batch size, error details, timestamp)
   - Move failed batch to DLQ
   - **Worker continues processing new batches**

2. **Dead Letter Queue**
   - Separate in-memory queue for failed batches
   - Max size configurable (default: 1000 batches)
   - FIFO eviction when full (log warning about oldest batch being dropped)
   - Persisted to disk periodically (optional, for durability across restarts)

3. **Retry Mechanism**
   - Background task that periodically (e.g., every 60s) attempts to drain DLQ
   - Exponential backoff per batch (1m, 5m, 15m, 1h)
   - After max attempts, log error and drop batch (with full trace IDs for recovery)

4. **Metrics & Monitoring**
   - Counter: `tracer_storage_failures_total`
   - Gauge: `tracer_dlq_size`
   - Counter: `tracer_traces_dropped_total`
   - Gauge: `tracer_queue_size`
   - Boolean: `tracer_worker_alive`

**Pros:**
- ✅ Worker never crashes - always processing new traces
- ✅ Failed traces get multiple retry opportunities
- ✅ Clear visibility into failure modes
- ✅ Can inspect/recover failed traces from DLQ
- ✅ Production-ready pattern (used in messaging systems)

**Cons:**
- ❌ Additional memory overhead for DLQ
- ❌ More complex implementation
- ❌ Traces in DLQ are not immediately available for querying

**Estimated Effort:** 2-3 days

---

### Option 2: Circuit Breaker Pattern

**Overview:** Detect repeated storage failures and "open" the circuit, temporarily dropping traces while logging errors. Periodically attempt to "close" the circuit.

**Architecture:**

```
Circuit States:
┌─────────┐  Failures > Threshold   ┌──────┐  Timeout    ┌──────────┐
│ CLOSED  │────────────────────────▶│ OPEN │────────────▶│ HALF_OPEN│
│(Normal) │                          │(Drop)│             │  (Test)  │
└─────────┘                          └──────┘             └──────────┘
     ▲                                                          │
     │                                     Success             │
     └──────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Circuit Breaker State Machine**
   - Tracks consecutive failures
   - Opens circuit after threshold (e.g., 5 consecutive failures)
   - Periodically transitions to HALF_OPEN to test recovery
   - Closes circuit after successful test

2. **Behavior by State**
   - **CLOSED:** Normal operation, all traces stored
   - **OPEN:** Drop all traces immediately, log once per minute
   - **HALF_OPEN:** Try one batch, close if success, reopen if failure

3. **Monitoring**
   - Alert when circuit opens (indicates storage unavailable)
   - Log trace drop count while circuit is open

**Pros:**
- ✅ Simple implementation
- ✅ Fast fail - no wasted retry attempts during outages
- ✅ Worker never crashes

**Cons:**
- ❌ **Traces dropped during outages** (no recovery)
- ❌ No buffering - data loss is permanent
- ❌ Less granular than DLQ approach

**Estimated Effort:** 1 day

---

### Option 3: Worker Health Monitor + Auto-Restart

**Overview:** Keep current retry behavior but add a health monitor that detects worker death and automatically restarts it.

**Architecture:**

```
┌──────────────────┐         ┌─────────────────┐
│  Health Monitor  │────────▶│  Worker Thread  │
│  (Watchdog)      │ Restart │  (Processing)   │
└──────────────────┘         └─────────────────┘
         │                            │
         │ Heartbeat                  │
         │◀───────────────────────────┘
         │
         ▼ No heartbeat for 30s
    [Restart Worker]
```

**Key Components:**

1. **Heartbeat Mechanism**
   - Worker updates timestamp after each batch
   - Health monitor checks heartbeat every 10s
   - If no update for 30s, assume worker dead

2. **Worker Restart**
   - Kill existing worker thread/loop
   - Create new worker thread with fresh event loop
   - Log restart event with reason

3. **Queue Preservation**
   - Existing queue survives restart
   - In-flight batch is lost (acceptable)

**Pros:**
- ✅ Minimal code changes
- ✅ Worker automatically recovers
- ✅ Simple mental model

**Cons:**
- ❌ Still loses in-flight batch on crash
- ❌ Doesn't prevent crashes, only recovers from them
- ❌ No visibility into why worker crashed
- ❌ Restart overhead (thread creation, event loop setup)

**Estimated Effort:** 1-2 days

---

## Recommendation: Hybrid Approach (Option 1 + Worker Health)

Combine **Option 1 (DLQ)** with a simplified health check from **Option 3**.

### Why This Combination?

1. **DLQ handles storage failures gracefully** - Primary problem solved
2. **Health check catches unexpected crashes** - Defense in depth
3. **Complete observability** - Know exactly what's happening
4. **Zero trace loss** - Failed traces go to DLQ, not dropped

### Implementation Phases

#### Phase 1: Core Resilience (Week 1)
**Goal:** Worker never crashes due to storage failures

**Changes to `rllm/sdk/tracing.py`:**

1. **Add DLQ to `__init__`:**
   ```python
   self._dead_letter_queue = queue.Queue(maxsize=1000)
   self._dlq_retry_task = None
   ```

2. **Wrap storage in `_async_worker`:**
   ```python
   async def _async_worker(self):
       while True:
           try:
               # ... batch collection ...
               if len(batch) == 1:
                   await self._store_trace_with_retry(batch[0])
               else:
                   await self._store_batch_with_retry(batch)
               # ... task_done ...
           except Exception as exc:
               # Log detailed error
               logger.error(f"Failed to store batch after retries: {exc}",
                          extra={"batch_size": len(batch), "error_type": type(exc).__name__})
               # Move to DLQ
               self._move_to_dlq(batch)
               # Mark as done so queue.join() doesn't hang
               for _ in batch:
                   self._trace_queue.task_done()
   ```

3. **Add DLQ retry task:**
   ```python
   async def _dlq_retry_worker(self):
       """Periodically retry failed batches from DLQ."""
       while not self._shutdown:
           await asyncio.sleep(60)  # Retry every 60s
           # Attempt to drain DLQ...
   ```

4. **Add metrics/counters:**
   ```python
   self._storage_failures = 0
   self._traces_dropped = 0
   self._dlq_size_metric = 0
   ```

**Testing:**
- Simulate storage failures (invalid endpoint, network timeout)
- Verify worker continues processing new traces
- Verify DLQ accumulates failed batches
- Verify DLQ drains when storage recovers

#### Phase 2: Health Monitoring (Week 2)
**Goal:** Visibility and auto-recovery from unexpected failures

**Changes to `rllm/sdk/tracing.py`:**

1. **Add health tracking:**
   ```python
   self._last_heartbeat = time.time()
   self._health_monitor_task = None
   ```

2. **Update heartbeat in worker:**
   ```python
   async def _async_worker(self):
       while True:
           # ... batch processing ...
           self._last_heartbeat = time.time()
   ```

3. **Add health monitor:**
   ```python
   async def _health_monitor(self):
       """Check worker health and restart if needed."""
       while not self._shutdown:
           await asyncio.sleep(10)
           if time.time() - self._last_heartbeat > 30:
               logger.critical("Worker heartbeat timeout! Restarting...")
               self._restart_worker()
   ```

**Testing:**
- Inject exception that bypasses DLQ handler
- Verify health monitor detects and restarts worker

#### Phase 3: Observability (Week 2-3)
**Goal:** Operators can monitor tracer health

**Changes to `scripts/litellm_proxy_server.py`:**

1. **Add tracer health endpoint:**
   ```python
   @litellm_app.get("/admin/tracer-health")
   async def tracer_health():
       if not runtime._tracer:
           return {"status": "disabled"}
       return {
           "status": "healthy" if tracer.is_healthy() else "unhealthy",
           "queue_size": tracer.queue_size(),
           "dlq_size": tracer.dlq_size(),
           "storage_failures": tracer.storage_failures,
           "traces_dropped": tracer.traces_dropped,
           "last_heartbeat_ago_seconds": time.time() - tracer.last_heartbeat
       }
   ```

**Testing:**
- Query health endpoint during normal operation
- Query during simulated storage failure
- Verify metrics are accurate

#### Phase 4: DLQ Persistence (Optional, Week 3-4)
**Goal:** Failed traces survive proxy restarts

**Implementation:**
- Periodically (every 5m) flush DLQ to disk (JSON file)
- On startup, load DLQ from disk if exists
- Allows recovery of failed traces across restarts

---

## Configuration Options

Add to `LLMTracer.__init__`:

```python
def __init__(
    self,
    # ... existing params ...
    max_dlq_size: int = 1000,
    dlq_retry_interval: float = 60.0,
    enable_dlq_persistence: bool = False,
    dlq_persistence_path: str | None = None,
    health_check_interval: float = 10.0,
    heartbeat_timeout: float = 30.0,
):
```

---

## Monitoring & Alerts

### Key Metrics to Track

1. **tracer_storage_failures_total** (counter)
   - Alert threshold: > 10 in 5 minutes
   - Action: Check context store health

2. **tracer_dlq_size** (gauge)
   - Alert threshold: > 500
   - Action: Investigate storage issues

3. **tracer_traces_dropped_total** (counter)
   - Alert threshold: > 0
   - Action: **Critical** - data loss occurring

4. **tracer_worker_alive** (boolean)
   - Alert threshold: false
   - Action: Worker crashed, check logs

5. **tracer_queue_size** (gauge)
   - Alert threshold: > 8000 (80% full)
   - Action: Worker falling behind or dead

### Log Messages

**Normal operation:**
```
[INFO] LLMTracer: Stored batch 123 with 8 items (queue: 45, DLQ: 0)
```

**Storage failure:**
```
[ERROR] LLMTracer: Failed to store batch after 3 retries: ConnectionError
[WARN] LLMTracer: Moved batch (8 traces) to DLQ (DLQ size: 12)
```

**DLQ recovery:**
```
[INFO] LLMTracer: DLQ retry succeeded, removed batch (8 traces, DLQ size: 4)
```

**Data loss:**
```
[CRITICAL] LLMTracer: DLQ full! Dropping batch (8 traces: tr_abc123, tr_def456, ...)
```

---

## Testing Strategy

### Unit Tests

1. **Test worker resilience:**
   - Mock context store to raise exceptions
   - Verify worker continues after failures
   - Verify DLQ receives failed batches

2. **Test DLQ retry:**
   - Fill DLQ with failed batches
   - Mock context store to succeed
   - Verify DLQ drains

3. **Test health monitor:**
   - Simulate worker hang (infinite sleep)
   - Verify health monitor detects and restarts

### Integration Tests

1. **End-to-end with simulated outage:**
   - Start proxy with tracer
   - Generate traces
   - Stop context store
   - Verify traces go to DLQ
   - Restart context store
   - Verify DLQ drains and traces appear

2. **Load test:**
   - Generate 100k traces
   - Simulate intermittent storage failures (20% failure rate)
   - Verify no worker crashes
   - Verify all traces eventually stored

---

## Rollback Plan

If issues are discovered in production:

1. **Immediate:** Deploy with `max_dlq_size=0` (disables DLQ, reverts to current behavior)
2. **Feature flag:** Add environment variable `RLLM_TRACER_DLQ_ENABLED=false`
3. **Monitoring:** Watch for increased memory usage (DLQ retention)

---

## Migration Path

### Breaking Changes
None - all changes are internal to LLMTracer

### Configuration Changes
New optional parameters with safe defaults:
- `max_dlq_size=1000` (reasonable default)
- `dlq_retry_interval=60.0` (1 minute)
- `enable_dlq_persistence=False` (opt-in)

### Deployment Steps

1. Deploy updated `rllm` package to staging
2. Run load tests with simulated failures
3. Monitor metrics for 24 hours
4. Deploy to production with gradual rollout (10% → 50% → 100%)
5. Enable DLQ persistence after 1 week of stable operation

---

## Alternative Approaches Considered

### 4. Sync Logging (No Background Worker)
**Rejected:** Would block LLM calls, unacceptable latency impact

### 5. Write-Ahead Log (WAL)
**Rejected:** Over-engineered for the use case, DLQ simpler

### 6. External Queue (Redis, RabbitMQ)
**Rejected:** Adds external dependency, not needed for scale

---

## Success Metrics

**Phase 1 (Core Resilience):**
- ✅ Zero worker crashes in 1 week of production traffic
- ✅ Storage failures handled gracefully (moved to DLQ)

**Phase 2 (Health Monitoring):**
- ✅ Worker auto-restarts within 30s if unexpected crash
- ✅ No gaps in trace coverage > 1 minute

**Phase 3 (Observability):**
- ✅ Operators can view tracer health via `/admin/tracer-health`
- ✅ Alerts fire correctly for degraded states

**Phase 4 (DLQ Persistence):**
- ✅ Failed traces survive proxy restarts
- ✅ Recovery after multi-hour context store outages

---

## Open Questions

1. **DLQ size:** Is 1000 batches enough? (At 8 traces/batch = 8000 traces)
   - **Answer:** Monitor in production, increase if needed

2. **Retry strategy:** Should we use exponential backoff per-batch or global interval?
   - **Recommendation:** Start with global interval (simpler), add per-batch if needed

3. **Persistence format:** JSON or binary (pickle)?
   - **Recommendation:** JSON for debuggability

4. **Max DLQ retention:** Should old batches expire after N hours?
   - **Recommendation:** Yes, add TTL of 24 hours to prevent unbounded growth

---

## References

- **Current Implementation:** `rllm/sdk/tracing.py:160-201`
- **Related Issue:** Worker crashes on storage failures
- **Similar Patterns:**
  - AWS SQS Dead Letter Queues
  - Kafka error handling
  - Celery task retry mechanisms
