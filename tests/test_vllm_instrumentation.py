"""Tests for vLLM instrumentation."""

from rllm.engine.vllm_instrumentation import (
    get_vllm_token_ids_support,
    instrument_vllm,
    is_vllm_instrumented,
    uninstrument_vllm,
)


def test_vllm_availability():
    """Test vLLM availability detection."""
    support = get_vllm_token_ids_support()
    assert support in ["native", "instrumented", "none", "unavailable"]
    print(f"vLLM token IDs support: {support}")


def test_instrument_vllm():
    """Test vLLM instrumentation."""
    # Try to instrument
    result = instrument_vllm()

    if result:
        # Instrumentation succeeded
        assert is_vllm_instrumented()
        print("✅ vLLM instrumented successfully")

        # Try to instrument again (should warn)
        result2 = instrument_vllm()
        assert not result2  # Should return False (already instrumented)

        # Uninstrument
        uninstrument_result = uninstrument_vllm()
        assert uninstrument_result
        assert not is_vllm_instrumented()
        print("✅ vLLM uninstrumented successfully")
    else:
        # Instrumentation not needed or failed
        support = get_vllm_token_ids_support()
        if support == "native":
            print("ℹ️  vLLM >= 0.10.2 detected, instrumentation not needed")
        elif support == "unavailable":
            print("ℹ️  vLLM not available, skipping instrumentation test")
        else:
            print("⚠️  Instrumentation failed")


def test_force_instrument():
    """Test force instrumentation."""
    # Force instrumentation even if vLLM >= 0.10.2
    result = instrument_vllm(force=True)

    if result:
        assert is_vllm_instrumented()
        print("✅ Force instrumentation successful")

        # Clean up
        uninstrument_vllm()
    else:
        support = get_vllm_token_ids_support()
        if support == "unavailable":
            print("ℹ️  vLLM not available, skipping force instrumentation test")


def test_multiple_instrument_calls():
    """Test multiple instrumentation calls."""
    # First call
    result1 = instrument_vllm()

    if result1:
        # Second call should return False
        result2 = instrument_vllm()
        assert not result2

        # Should still be instrumented
        assert is_vllm_instrumented()

        # Clean up
        uninstrument_vllm()
        print("✅ Multiple instrumentation calls handled correctly")
    else:
        print("ℹ️  Instrumentation not applicable, skipping test")


def test_uninstrument_without_instrument():
    """Test uninstrumentation without prior instrumentation."""
    # Make sure not instrumented
    if is_vllm_instrumented():
        uninstrument_vllm()

    # Try to uninstrument
    result = uninstrument_vllm()

    # Should return False (nothing to uninstrument)
    assert not result
    print("✅ Uninstrumentation without prior instrumentation handled correctly")


if __name__ == "__main__":
    print("Testing vLLM instrumentation...")
    print("=" * 80)

    test_vllm_availability()
    print()

    test_instrument_vllm()
    print()

    test_force_instrument()
    print()

    test_multiple_instrument_calls()
    print()

    test_uninstrument_without_instrument()
    print()

    print("=" * 80)
    print("All tests completed!")
