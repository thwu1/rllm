# Supervised Fine-Tuning with the rllm SDK (Fireworks)

We are going to use SFT together with the rllm SDK to auto-log the LLM calls and train the Qwen/Qwen2-7B-Instruct model.

## 1. Launch the Teacher Model

Start vLLM with Qwen/Qwen2.5-7B-Instruct, which acts as the teacher:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

## 2. Generate Trajectories

Run `run_math_teacher.py`. This uses Qwen/Qwen2.5-7B-Instruct to generate trajectories for GSM8K problems (only the first 1,000 train problems).

You should see an accuracy around 0.33 on the test set for the teacher model, and 0.0 for the student model (because the student model would not output the answer in `\boxed{}` format).

## 3. Prepare the Dataset

Once the generation finishes, run `get_dataset.py` to convert the traces into a JSONL file that Fireworks AI accepts.

## 4. Fine-Tune on Fireworks AI

Upload the JSONL to the Fireworks AI dashboard as a dataset, and start a fine-tune job.

The training job should finish in about 10 minutes. Afterward, deploy the fine-tuned model.

## 5. Evaluate the Fine-Tuned Model

Fill in your Fireworks API key in `eval_trained_model.py`, along with the deployed model name, then run the evaluation. You should see that the SFTed model reaches around 0.33 accuracy on the test set.
