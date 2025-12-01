import base64
import io
from io import BytesIO

from PIL import Image

from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction, math_reward_fn
from rllm.sdk import get_chat_client_async
from rllm.workflows.workflow import Workflow


def image_to_jpg_base64_url(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 encoded image URL."""
    # Convert any non-RGB mode to RGB (handles RGBA, LA, P, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


class Geo3KWorkflowSdk(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.reward_fn: RewardFunction = reward_function or math_reward_fn
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
        self.model = "Qwen/Qwen3-VL-2B-Instruct"

    async def run(self, task: dict, uid: str, **kwargs) -> float:
        self.reset(task, uid)

        question = task.get("question")
        image = task.get("image", task.get("images", None))
        if isinstance(image, list) and len(image) > 0:
            image = image[0]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        assert isinstance(image, Image.Image) or isinstance(image, str) or image is None, f"Image must be a PIL.Image.Image or a string, but got {type(image)}"

        if image is None:
            return []

        base64_url = image_to_jpg_base64_url(image)
        messages = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": base64_url}}]}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_tokens=2048,
        )
        solution = response.choices[0].message.content
        reward = self.reward_fn(task, solution).reward

        return reward
