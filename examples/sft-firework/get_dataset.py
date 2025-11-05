import json

from rllm.sdk import RLLMClient

rllm_client = RLLMClient(
    base_url="http://localhost:30000/v1",
    api_key="None",
    project="sft-firework",
    cs_endpoint="http://localhost:8000",
    cs_api_key="your-api-key-here",
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
traces = rllm_client.get_session_traces(session_id=f"batch-{model_name}-first1000")


def filter_key(message, zero_weight=True):
    for key in list(message.keys()):
        if key not in ["role", "content"]:
            del message[key]
    role = message.get("role", None)
    if zero_weight:
        if role:
            if role == "assistant":
                message["weight"] = 0
    return message


def convert_trace_to_row(trace):
    messages = trace["input"]["messages"]
    output = trace["output"]["choices"][0]["message"]

    new_message = [filter_key(message, zero_weight=True) for message in messages]
    new_message.append(filter_key(output, zero_weight=False))
    return {"messages": new_message}


# write to a jsonl file
with open("qwen25-7b-intruct-first1000.jsonl", "w") as f:
    for trace in traces:
        f.write(json.dumps(convert_trace_to_row(trace)) + "\n")
