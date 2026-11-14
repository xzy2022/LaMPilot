"""固定官方 Demo ID 的小规模验证；不代表论文独立测试集评测。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parents[1] / "experiment"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_key(env_file):
    """只读取指定 .env 中的配置；已有进程环境变量优先。绝不向外输出密钥。"""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if separator:
            name = name.strip()
            value = value.strip().strip("\"'")
            if name and name not in os.environ:
                os.environ[name] = value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "api"], default="offline")
    parser.add_argument("--dataset", choices=["demo", "table3", "all"], default="table3",
                        help="评测数据集：demo 为 12 个 Demo 样本；table3 为 Table 3 默认的前 98 个样本（dec_abs_dis15）；all 为全量 4900 个样本")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--limit", type=int, default=98)
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="父项目 experiment/ 下的实验输出目录")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--episode-timeout", type=int, default=240)
    parser.add_argument("--worker", metavar="IID", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_items(dataset_name, limit):
    if dataset_name == "demo":
        from projects.lampilot.dt.dbl import DbLv1DemoDataset
        dataset = DbLv1DemoDataset("projects/lampilot/configs/DbLv1")
    elif dataset_name in ("table3", "all"):
        from projects.lampilot.dt.dbl import DbLv1Dataset
        dataset = DbLv1Dataset("projects/lampilot/configs/DbLv1")
    else:
        raise ValueError(f"未知数据集：{dataset_name}")
    return list(dataset)[:limit]


def contract(args, items):
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    hashes = {}
    for filename in tracked:
        if filename.endswith(".py") or filename.startswith("projects/lampilot/prompts/"):
            hashes[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    # 初次提交前执行验证时，也纳入新入口的实际代码。
    for path in (ROOT / "reproduction").glob("*.py"):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = {
        "mode": args.mode, "model": args.model if args.mode == "api" else None,
        "dataset": args.dataset,
        "prompting": "zero-shot" if args.zero_shot else "few-shot",
        "selection": f"DbLv1_{args.dataset}_{len(items)}items",
        "episode_timeout_seconds": args.episode_timeout,
        "items": items, "source_hashes": hashes,
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return {
        "signature": signature, "settings": settings,
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in [
            "numpy", "gymnasium", "pygame", "langchain", "langchain-openai", "openai",
        ]},
        "scope": "small smoke check, not the paper's held-out test set",
    }


def episode(args, item):
    from PIL import Image
    from projects.lampilot.dt.cg_agent import CodeGenerationAgent
    from projects.lampilot.dt.vehicle_dt import CtrlVDT
    from projects.lampilot.evaluator import get_evaluator_class
    from projects.lampilot.utils.result import create_result_dict

    iid = item["id"]
    output = args.output / iid
    started = time.perf_counter()
    evaluator = None
    phase = "environment"
    try:
        evaluator = get_evaluator_class(item["sample"]["eval"]["type"])(
            config=item["sample"], show_window=False, record_video=False, wait_time=0,
        )
        # 只修改渲染配置。SDL dummy 会让上游 EnvViewer 禁用绘制并输出全黑图片。
        evaluator.env.unwrapped.configure({"offscreen_rendering": True, "real_time_rendering": False})
        context = evaluator.get_context_info()
        controller = CtrlVDT()
        controller.reset(ego_vehicle=evaluator.env.unwrapped.vehicle)
        output.mkdir(parents=True, exist_ok=True)
        frame = evaluator.env.render()
        if frame is None or frame.std() == 0:
            raise RuntimeError("离屏渲染输出空白图像，请检查 SDL_VIDEODRIVER 是否设成 dummy。")
        Image.fromarray(frame).save(output / "initial.png")
        evaluator.env.unwrapped.enable_auto_render = False
        if args.mode == "api":
            phase = "model_request"
            if "deepseek" in args.model.lower():
                from reproduction.deepseek_agent import DeepSeekAgent
                agent = DeepSeekAgent(args.model, zero_shot=args.zero_shot)
            else:
                from reproduction.gpt_agent import GPTAgent
                agent = GPTAgent(args.model, zero_shot=args.zero_shot)
            agent.reset(item["command"], context)
            write_json(output / "prompt.json", [
                {"role": "system", "content": agent.messages[0].content},
                {"role": "user", "content": agent.messages[1].content},
            ])
            try:
                policy = agent.step()
            finally:
                write_json(output / "response.json", agent.last_response)
        else:
            # 唯一人工目标策略只用于验证“代码→控制→评分”链路，不计为 LLM 结果。
            if iid == "dec_abs_speed10_s0_c0":
                response = "```python\ndef offline_speed_check():\n    set_target_speed(10)\n```"
            else:
                response = "```python\ndef offline_autopilot_check():\n    yield autopilot()\n```"
            parsed = CodeGenerationAgent.process_ai_message(response)
            policy = {"reused_code": "", "new_code": parsed["program_code"] + "\n" + parsed["exec_code"]}
        write_json(output / "policy.json", policy)
        phase = "simulation"
        controller.execute(policy)
        while not evaluator.ended:
            evaluator.step(controller)
        frame = evaluator.env.render()
        if frame is None or frame.std() == 0:
            raise RuntimeError("结束状态的离屏渲染输出空白图像。")
        Image.fromarray(frame).save(output / "final.png")
        evaluator.env.unwrapped.enable_auto_render = False
        result = create_result_dict(iid, evaluator, show=False, code=policy, command=item["command"], context_info=context)
        result.update({
            "status": "completed", "mode": args.mode, "seed": item["sample"]["seed"],
            "simulation_seconds": evaluator.overall_time,
            "wall_seconds": time.perf_counter() - started,
            "offline_handwritten_speed_check": args.mode == "offline" and iid == "dec_abs_speed10_s0_c0",
        })
    except Exception as exc:
        error = str(exc)
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            error = error.replace(key, "[REDACTED]")
        result = {
            "iid": iid, "status": "error", "phase": phase,
            "error_type": type(exc).__name__, "error": error,
            "wall_seconds": time.perf_counter() - started,
        }
        # 不输出原始请求对象、环境变量或密钥。
        print(f"{type(exc).__name__}: {error}")
    finally:
        if evaluator is not None:
            evaluator.close()
    write_json(output / "result.json", result)
    return result


def summary(output, items, args=None):
    results = []
    usage = {}
    for item in items:
        folder = output / item["id"]
        if not (folder / "result.json").exists():
            continue
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        log = (folder / "run.log").read_text(encoding="utf-8", errors="replace")
        result["policy_runtime_error_count"] = log.count("Error in execute:") + log.count("Error in act:")
        write_json(folder / "result.json", result)
        results.append(result)
        if (folder / "response.json").exists():
            response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
            for key, value in response.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
                elif isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, (int, float)):
                            flat_key = f"{key}_{subkey}"
                            usage[flat_key] = usage.get(flat_key, 0) + subval
    evaluated = [r for r in results if r["status"] == "completed"]
    successful = [r for r in evaluated if r["success"]]
    mean_success_score = sum(r["overall_score"] for r in successful) / len(successful) if successful else 0
    success_rate = len(successful) / len(evaluated) if evaluated else None
    collision_rate = sum(r["collision"] for r in evaluated) / len(evaluated) if evaluated else None
    scope_str = (
        f"LaMPilot evaluation (dataset={getattr(args, 'dataset', 'unknown')}, model={getattr(args, 'model', 'unknown')}, limit={len(items)})"
        if args else f"LaMPilot {len(items)} items evaluation"
    )
    report = {
        "planned": len(items), "attempted": len(results), "completed": len(evaluated),
        "errors": len(results) - len(evaluated), "successes": len(successful),
        "collisions": sum(r["collision"] for r in evaluated),
        "completion_rate_among_simulated_episodes": success_rate,
        "collision_rate_among_simulated_episodes": collision_rate,
        "driving_score_among_simulated_episodes": success_rate * mean_success_score - collision_rate * 500 if evaluated else None,
        "policy_runtime_error_count": sum(r["policy_runtime_error_count"] for r in results),
        "observed_token_usage": usage,
        "token_usage_note": "Returned usage only; provider-internal retries may add unobserved cost.",
        "scope": scope_str,
    }
    write_json(output / "results.json", results)
    write_json(output / "summary.json", report)
    return report


def main():
    args = arguments()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    args.output = args.output.resolve()
    if EXPERIMENT_ROOT.resolve() not in args.output.parents:
        raise SystemExit("按实验通用原则，--output 必须指向父项目 experiment/ 下的目录。")
    if args.limit < 1 or args.episode_timeout < 1:
        raise SystemExit("limit 和 episode-timeout 必须大于 0。")
    load_key(args.env_file.resolve())
    if args.mode == "api":
        if "deepseek" in args.model.lower():
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise SystemExit("尚未配置 DEEPSEEK_API_KEY；可填写仓库根目录 .env 或设置进程环境变量。")
        else:
            if not (os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")):
                raise SystemExit("尚未配置 GPT_API_KEY 或 OPENAI_API_KEY；可填写仓库根目录 .env 或设置进程环境变量。")
    items = load_items(args.dataset, args.limit)
    if args.worker:
        item = next(item for item in items if item["id"] == args.worker)
        return 0 if episode(args, item)["status"] == "completed" else 1
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = contract(args, items)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["signature"] != manifest["signature"]:
            raise SystemExit("已有目录的配置、代码或样本与本次不同；请用 --output 指定新目录。")
    else:
        write_json(manifest_path, manifest)
    for item in items:
        folder = args.output / item["id"]
        result_file = folder / "result.json"
        if result_file.exists():
            print(f"[skip] {item['id']} 已有结果（含失败结果），不会重复付费调用。", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "reproduction.run_small", "--worker", item["id"],
            "--mode", args.mode, "--model", args.model, "--dataset", args.dataset, "--limit", str(args.limit),
            "--output", str(args.output), "--env-file", str(args.env_file.resolve()),
        ]
        if args.zero_shot:
            command.append("--zero-shot")
        print(f"[run] {item['id']} mode={args.mode}", flush=True)
        timed_out = False
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.episode_timeout, cwd=ROOT)
            except subprocess.TimeoutExpired:
                timed_out = True
                process = None
        if timed_out or not result_file.exists():
            write_json(result_file, {
                "iid": item["id"], "status": "error",
                "phase": "worker", "error_type": "TimeoutExpired" if timed_out else "WorkerExit",
                "error": "Episode wall-time limit exceeded" if timed_out else f"Worker exit {process.returncode}",
            })
        result = json.loads(result_file.read_text(encoding="utf-8"))
        print(f"[result] {item['id']} status={result['status']} success={result.get('success')}", flush=True)
        summary(args.output, items, args)
        if result.get("phase") == "model_request" and result.get("error_type") in {
            "AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError",
            "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
            "APIStatusError",
        }:
            print("API 服务请求失败，停止后续调用；详见该场景 result.json。", flush=True)
            break
    report = summary(args.output, items, args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["errors"] == 0 and report["attempted"] == report["planned"] else 1


if __name__ == "__main__":
    sys.exit(main())
