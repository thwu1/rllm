# OTelSession Test Suite

Comprehensive test suite for the OpenTelemetry-based distributed session tracking implementation.

## Test Files

### `test_otel_session.py` (~300 lines)
**Basic unit tests for core OTelSession functionality**

- **TestOTelSessionBasics**: Session creation, naming, metadata, storage defaults
- **TestOTelSessionContext**: Context manager behavior, active session tracking
- **TestOTelSessionNesting**: Nested sessions, metadata inheritance, UID chains
- **TestOTelSessionSerialization**: to_otel_context/from_otel_context for cross-process
- **TestOTelSessionStorage**: Integration with InMemoryStorage and SqliteSessionStorage
- **TestOTelSessionHelpers**: Helper functions (get_current_otel_session, etc.)
- **TestOTelSessionRepr**: String representation

**Coverage**: Core OTelSession class functionality

---

### `test_metadata_assembly.py` (~100 lines)
**Tests for dual-mode metadata assembly**

- **TestDualModeMetadataAssembly**: Auto-detection of session type
- **TestMetadataAssemblyEdgeCases**: Extra metadata, overrides, deduplication

**Coverage**: metadata_slug.py dual-mode assembly

---

### `test_otel_integration.py` (~450 lines)
**Integration tests for HTTP propagation and middleware**

- **TestHTTPInstrumentation**: init_otel_distributed_tracing(), auto-init
- **TestBaggagePropagation**: OTel baggage setting and retrieval
- **TestMiddlewareBaggageExtraction**: Middleware extracting from HTTP headers
- **TestMetadataAssemblyWithOTel**: Integration with metadata assembly
- **TestConcurrentSessions**: ContextVarSession and OTelSession coexistence
- **TestErrorHandling**: Graceful degradation without OTel
- **TestStorageIntegration**: Storage backends with OTelSession

**Coverage**: HTTP propagation, middleware, baggage extraction

---

### `test_otel_ray.py` (~400 lines)
**Tests for Ray integration and cross-process serialization**

- **TestRayEntrypointDecorator**: @ray_entrypoint decorator functionality
- **TestCrossProcessSerialization**: Serialization roundtrips, chain preservation
- **TestRayWorkerSimulation**: Simulated Ray worker scenarios
- **TestMultiprocessingScenarios**: Shared storage, nested workers
- **TestEdgeCases**: Empty metadata, None values, special characters, large metadata

**Coverage**: Ray integration, serialization for distributed scenarios

---

### `test_otel_performance.py` (~400 lines)
**Performance and stress tests**

- **TestPerformance**: Session creation overhead, enter/exit cycles, serialization speed
- **TestConcurrency**: Concurrent session creation, nested sessions from threads
- **TestStress**: Many sequential sessions, rapid cycling, large stacks
- **TestMemoryUsage**: Memory cleanup, garbage collection
- **TestScalability**: Concurrent sessions with storage, deep nesting
- **TestEdgeCasePerformance**: Minimal vs complex metadata

**Coverage**: Performance characteristics, scalability, concurrent usage

---

### `test_otel_compatibility.py` (~450 lines)
**Compatibility and real-world scenario tests**

- **TestContextVarSessionCompatibility**: API parity with ContextVarSession
- **TestRealWorldHTTPScenarios**: HTTP microservice simulations
- **TestErrorHandlingComprehensive**: Exception handling, cleanup
- **TestBackwardsCompatibility**: Drop-in replacement patterns
- **TestSessionProtocolCompliance**: SessionProtocol implementation
- **TestMetadataSlugIntegration**: Slug encoding/decoding integration
- **TestRealWorldUsagePatterns**: Training loops, A/B testing, user tracking
- **TestDocumentationExamples**: All documentation examples work

**Coverage**: Backwards compatibility, real-world usage, protocol compliance

---

## Running Tests

### Run all OTelSession tests
```bash
pytest tests/sdk/ -v
```

### Run specific test file
```bash
pytest tests/sdk/test_otel_session.py -v
pytest tests/sdk/test_otel_integration.py -v
pytest tests/sdk/test_otel_ray.py -v
pytest tests/sdk/test_otel_performance.py -v
pytest tests/sdk/test_otel_compatibility.py -v
pytest tests/sdk/test_metadata_assembly.py -v
```

### Run specific test class
```bash
pytest tests/sdk/test_otel_session.py::TestOTelSessionBasics -v
pytest tests/sdk/test_otel_integration.py::TestHTTPInstrumentation -v
```

### Run specific test
```bash
pytest tests/sdk/test_otel_session.py::TestOTelSessionBasics::test_session_creation -v
```

### Run with coverage
```bash
pytest tests/sdk/ --cov=rllm.sdk.session.otel --cov-report=html
```

### Run performance tests only
```bash
pytest tests/sdk/test_otel_performance.py -v
```

### Run tests that don't require mocking
```bash
pytest tests/sdk/ -v -m "not requires_otel"
```

---

## Test Coverage Summary

| Component | Test File | Lines | Coverage |
|-----------|-----------|-------|----------|
| OTelSession core | test_otel_session.py | 300 | ✅ Core API |
| Metadata assembly | test_metadata_assembly.py | 100 | ✅ Dual-mode |
| HTTP/Middleware | test_otel_integration.py | 450 | ✅ Baggage |
| Ray integration | test_otel_ray.py | 400 | ✅ Serialization |
| Performance | test_otel_performance.py | 400 | ✅ Stress tests |
| Compatibility | test_otel_compatibility.py | 450 | ✅ Real-world |

**Total**: ~2,100 lines of test code

---

## Test Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (OTel libraries)
- Fast execution, no external services required

### Integration Tests
- Test component interactions
- Mock HTTP requests/responses
- Test middleware and metadata assembly

### Performance Tests
- Measure overhead and scalability
- Concurrent and parallel execution
- Memory usage and cleanup

### Compatibility Tests
- Ensure API parity with ContextVarSession
- Test real-world usage patterns
- Verify backwards compatibility

---

## Mocking Strategy

Most tests mock OpenTelemetry components to:
1. Avoid requiring full OTel installation
2. Enable fast test execution
3. Isolate OTelSession logic from OTel internals
4. Test error handling when OTel unavailable

Key mocks:
- `rllm.sdk.session.otel._ensure_instrumentation`
- `rllm.sdk.session.otel.baggage`
- `rllm.sdk.session.otel.context`
- `rllm.sdk.session.otel.trace`
- `rllm.sdk.session.otel.RequestsInstrumentor`
- `rllm.sdk.session.otel.HTTPXClientInstrumentor`

---

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run OTelSession tests
  run: |
    pip install pytest pytest-asyncio
    pytest tests/sdk/ -v --tb=short
```

All tests pass without requiring:
- Actual HTTP servers
- Real Ray clusters
- Full OpenTelemetry installation (mocked)
- External services

---

## Adding New Tests

When adding new tests:

1. **Choose the right file**:
   - Core functionality → `test_otel_session.py`
   - HTTP/middleware → `test_otel_integration.py`
   - Ray/serialization → `test_otel_ray.py`
   - Performance → `test_otel_performance.py`
   - Compatibility → `test_otel_compatibility.py`

2. **Follow naming conventions**:
   - Test classes: `TestFeatureName`
   - Test methods: `test_specific_behavior`
   - Use descriptive names

3. **Use appropriate mocks**:
   - Always mock `_ensure_instrumentation` for OTelSession creation
   - Mock OTel libraries when testing baggage/context

4. **Add docstrings**:
   - Explain what the test validates
   - Document expected behavior

5. **Keep tests fast**:
   - Use in-memory storage when possible
   - Minimize sleep/wait times
   - Mock external dependencies

---

## Known Limitations

1. **HTTP instrumentation**: Tests mock the instrumentation calls, don't test actual HTTP propagation with real requests
2. **Ray integration**: Tests simulate Ray workers, don't test with actual Ray cluster
3. **OTel backend**: Tests don't send traces to actual OTel collectors
4. **Async tests**: Limited async test coverage (mostly sync patterns)

For end-to-end testing with real OTel infrastructure, see integration test documentation.

---

## Troubleshooting

### Tests fail with "ImportError: No module named 'opentelemetry'"
**Solution**: Tests should mock OTel imports. Check that `@patch` decorators are present.

### Tests hang or timeout
**Solution**: Check for missing mocks that might be trying to make real HTTP calls.

### Concurrent test failures
**Solution**: Ensure tests properly clean up sessions and storage between runs.

### SQLite locking errors
**Solution**: Use `:memory:` databases for tests, or unique temp files per test.

---

## Future Test Additions

Potential areas for additional test coverage:

- [ ] End-to-end tests with real OTel collectors
- [ ] Integration with real Ray clusters
- [ ] HTTP client instrumentation with real requests
- [ ] gRPC propagation tests
- [ ] Trace visualization and querying
- [ ] Performance benchmarking suite
- [ ] Load testing with production-like loads
- [ ] Multi-machine distributed scenarios
