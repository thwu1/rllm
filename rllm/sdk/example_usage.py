"""Example usage of RLLM SDK context manager with tracer."""

from rllm.sdk import RLLMClient


def simulate_agent_execution():
    """
    Example showing how to use the context manager to automatically
    propagate session_id and metadata through nested function calls.
    """

    client = RLLMClient()

    # Before: You had to pass session_id through every function
    # def run_agent(test_set, session_id):
    #     for task in test_set:
    #         result = solve_task(task, session_id)
    #
    # def solve_task(task, session_id):
    #     tracer.log_llm_call(..., session_id=session_id)

    # After: Just wrap with context manager
    def run_agent(test_set):
        """Run agent on test set - no session_id parameter needed!"""
        results = []
        for task in test_set:
            result = solve_task(task)
            results.append(result)
        return results

    def solve_task(task):
        """Solve single task - still no session_id parameter!"""
        # In real code, this would call:
        # tracer.log_llm_call(...)
        # And session_id would be automatically injected from context

        # Simulate getting context
        from rllm.sdk.context import get_current_metadata, get_current_session

        return {"task": task, "session_id": get_current_session(), "metadata": get_current_metadata()}

    # Example 1: Simple session
    print("Example 1: Simple session")
    with client.session("simple-session"):
        results = run_agent(["task1", "task2"])
        for r in results:
            print(f"  {r['task']}: session={r['session_id']}")

    # Example 2: Session with metadata
    print("\nExample 2: Session with metadata")
    with client.session("metadata-session", experiment="v1", model="gpt-4"):
        results = run_agent(["task3", "task4"])
        for r in results:
            print(f"  {r['task']}: session={r['session_id']}, meta={r['metadata']}")

    # Example 3: Multiple runs (auto-generated session IDs)
    print("\nExample 3: Multiple runs with auto-generated session IDs")
    for iteration in range(3):
        with client.session(iteration=iteration, run_type="benchmark"):
            results = run_agent([f"task_{iteration}"])
            for r in results:
                print(f"  Iteration {iteration}: session={r['session_id'][:20]}..., meta={r['metadata']}")

    # Example 4: Nested contexts
    print("\nExample 4: Nested contexts (metadata inheritance)")
    with client.session("outer", experiment="v1"):
        print("  Outer context")
        result = solve_task("outer_task")
        print(f"    {result['task']}: session={result['session_id']}, meta={result['metadata']}")

        with client.session("inner", task_type="math"):
            print("  Inner context (inherits experiment=v1)")
            result = solve_task("inner_task")
            print(f"    {result['task']}: session={result['session_id']}, meta={result['metadata']}")

        print("  Back to outer context")
        result = solve_task("outer_task_2")
        print(f"    {result['task']}: session={result['session_id']}, meta={result['metadata']}")


if __name__ == "__main__":
    simulate_agent_execution()
