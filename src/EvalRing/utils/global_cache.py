import os
import json
import sqlite3
import hashlib
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Literal, List

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

CacheMode = Literal["runs_only", "cache_file", "both", "none"]


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "n/a", "â€”"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _to_int(v):
    fv = _to_float(v)
    return int(fv) if fv is not None else None

def _read_all_cases_csv(csv_path: Path, ignore_errors: bool = False) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    normalized = []
    for r in rows:
        raw_class_scores = r.get("class_scores")
        parsed_class_scores = None
        if raw_class_scores:
            try:
                parsed_class_scores = json.loads(raw_class_scores)
            except Exception:
                parsed_class_scores = None

        normalized.append({
            "sample_id": r.get("sample_id", ""),
            "ground_truth": r.get("ground_truth", ""),
            "prediction": r.get("prediction", ""),
            "correct": _to_int(r.get("correct")) or 0,
            "prediction_confidence": _to_float(r.get("prediction_confidence")),
            "class_scores": parsed_class_scores,
            "ttft": _to_float(r.get("ttft")),
            "tps": _to_float(r.get("tps")),
            "total_time": _to_float(r.get("total_time")),
            "generation_time": _to_float(r.get("generation_time")),
            "prompt_tokens": _to_int(r.get("prompt_tokens")),
            "completion_tokens": _to_int(r.get("completion_tokens")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "error": (r.get("error") or "").strip() or None,
            "text": r.get("text", ""),
            "from_cache": True,
        })
        
    # Filter out errors
    filtered = []
    for case in normalized:
        if ignore_errors and (str(case.get("prediction", "")).strip().lower() == "error" or case.get("error")):
            continue
        filtered.append(case)
        
    return filtered

def _read_all_cases_jsonl(jsonl_path: Path, ignore_errors: bool = False) -> List[Dict[str, Any]]:
    cases = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        case = json.loads(line)
                        if ignore_errors and (str(case.get("prediction", "")).strip().lower() == "error" or case.get("error")):
                            continue
                        case["from_cache"] = True
                        cases.append(case)
                    except json.JSONDecodeError as e:
                        pass
    except Exception as e:
        return []
    return cases

def _read_all_cases_txt(txt_path: Path, ignore_errors: bool = False) -> List[Dict[str, Any]]:
    cases = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = content.split("-" * 40)
        for block in blocks[1:]:  # Skip header
            block = block.strip()
            if not block:
                continue
            case_data = {"from_cache": True}
            lines = block.split("\n")
            text_start = -1
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Sample ID :"):
                    case_data["sample_id"] = line.replace("Sample ID :", "").strip().split("[")[0].strip()
                elif line.startswith("GT"):
                    case_data["ground_truth"] = line.replace("GT", "").strip().lstrip(":").strip()
                elif line.startswith("Pred"):
                    case_data["prediction"] = line.replace("Pred", "").strip().lstrip(":").strip()
                elif line.startswith("Timing"):
                    timing_str = line.replace("Timing", "").strip().lstrip(":")
                    try:
                        if "TTFT=" in timing_str:
                            case_data["ttft"] = _to_float(timing_str.split("TTFT=")[1].split("s")[0])
                        if "Gen=" in timing_str:
                            case_data["generation_time"] = _to_float(timing_str.split("Gen=")[1].split("s")[0])
                        if "TPS=" in timing_str:
                            case_data["tps"] = _to_float(timing_str.split("TPS=")[1].split("|")[0])
                        if "Total=" in timing_str:
                            case_data["total_time"] = _to_float(timing_str.split("Total=")[1].split("s")[0])
                    except Exception:
                        pass
                elif line.startswith("Tokens"):
                    tokens_str = line.replace("Tokens", "").strip().lstrip(":")
                    try:
                        if "Prompt=" in tokens_str:
                            case_data["prompt_tokens"] = _to_int(tokens_str.split("Prompt=")[1].split("|")[0])
                        if "Completion=" in tokens_str:
                            case_data["completion_tokens"] = _to_int(tokens_str.split("Completion=")[1].split("|")[0])
                        if "Total=" in tokens_str:
                            case_data["total_tokens"] = _to_int(tokens_str.split("Total=")[1])
                    except Exception:
                        pass
                elif line.startswith("Confidence"):
                    case_data["prediction_confidence"] = _to_float(line.replace("Confidence", "").strip().lstrip(":"))
                elif line.startswith("ClassDist"):
                    try:
                        case_data["class_scores"] = json.loads(line.replace("ClassDist", "").strip().lstrip(":"))
                    except Exception:
                        pass
                elif line.startswith("Error"):
                    error_str = line.replace("Error", "").strip().lstrip(":")
                    case_data["error"] = error_str if error_str else None
                elif line.startswith("Text:"):
                    text_start = i + 1
                    break
            if text_start >= 0:
                case_data["text"] = "\n".join(lines[text_start:]).strip()
            else:
                case_data["text"] = ""
            for key in ["ground_truth", "prediction", "text"]:
                if key not in case_data:
                    case_data[key] = ""
            
            if ignore_errors and (str(case_data.get("prediction", "")).strip().lower() == "error" or case_data.get("error")):
                continue

            if "sample_id" in case_data:
                cases.append(case_data)
    except Exception as e:
        return []
    return cases

def _load_cached_cases(directory: Path, ignore_errors: bool = False) -> List[Dict[str, Any]]:
    if not directory.exists():
        return []
    jsonl_path = directory / "all_cases_partial.jsonl"
    if jsonl_path.exists():
        cases = _read_all_cases_jsonl(jsonl_path, ignore_errors)
        if cases: return cases
    txt_path = directory / "all_cases.txt"
    if txt_path.exists():
        cases = _read_all_cases_txt(txt_path, ignore_errors)
        if cases: return cases
    csv_path = directory / "all_cases.csv"
    if csv_path.exists():
        cases = _read_all_cases_csv(csv_path, ignore_errors)
        if cases: return cases
    return []

def _discover_cache_candidate_dirs(cache_dir: Path) -> List[Path]:
    candidates = set()
    if not cache_dir.exists():
        return []
        
    print(f"Scanning for cache candidates in {cache_dir}...", flush=True)
    
    def check_dir(d: Path):
        for cache_file_name in ("Meta.json", "all_cases_partial.jsonl", "all_cases.txt", "all_cases.csv"):
            if (d / cache_file_name).exists():
                candidates.add(d)
                break

    try:
        # Check cache_dir itself
        check_dir(cache_dir)
        
        # Check immediate children of cache_dir
        for item1 in cache_dir.iterdir():
            if item1.is_dir():
                check_dir(item1)
                
                # Check children of run_suite_*
                if item1.name.startswith("run_suite_"):
                    for item2 in item1.iterdir():
                        if item2.is_dir():
                            check_dir(item2)
    except OSError:
        pass
        
    print(f"Found {len(candidates)} candidate cache directories.", flush=True)
    return sorted(list(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

def _infer_model_name_from_cases(cases: List[Dict[str, Any]]) -> Optional[str]:
    for case in cases:
        model_name = case.get("model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return None

def load_legacy_runs(cache_dir: Path, model_name: str, agent_mode: str, base_class: str, ignore_errors: bool = False) -> Dict[str, Dict[str, Any]]:
    """Discover and load relevant cached cases from legacy runs."""
    combined_cases = {}
    model_suffix = model_name.split("/")[-1].lower() if model_name else ""
    candidates = _discover_cache_candidate_dirs(cache_dir)
    for d in tqdm(candidates, desc=f"Loading cache for {model_suffix or 'model'}"):
        meta_path = d / "Meta.json"
        meta_matched = False
        skip_candidate = False
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                m_cfg = meta.get("model_config", {})
                if (m_cfg.get("model_name") == model_name
                    and m_cfg.get("agent_mode") == agent_mode
                    and m_cfg.get("base_class") == base_class):
                    meta_matched = True
                else:
                    skip_candidate = True
            except Exception:
                pass
        
        if skip_candidate:
            continue
        
        loaded_cases = _load_cached_cases(d, ignore_errors)
        if not loaded_cases:
            continue
            
        if not meta_matched:
            inferred_model = _infer_model_name_from_cases(loaded_cases)
            if inferred_model:
                if inferred_model != model_name:
                    continue
            elif model_suffix and model_suffix not in d.name.lower():
                continue
            if agent_mode and agent_mode not in d.name:
                continue
            if base_class and base_class.lower() not in d.name.lower():
                continue
        
        for case in loaded_cases:
            sid = str(case.get("sample_id", ""))
            if sid and sid not in combined_cases:
                combined_cases[sid] = case
                
    return combined_cases

class GlobalCache:
    """
    A unified caching system for EvalRing LLM responses and evaluations.
    Implements a thread/process-safe SQLite backend within _EvalRing/Cache.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, mode: CacheMode = "both", workspace_root: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            if mode:
                self.mode = mode
            return

        self._initialized = True
        self.mode = mode

        if workspace_root is None:
            workspace_root = os.getcwd()

        self.cache_dir = Path(workspace_root) / "_EvalRing" / "Cache"
        self.base_runs_dir = Path(workspace_root) / "_EvalRing"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "evalring_cache.sqlite"
        self._legacy_cache_loaded = False
        self._legacy_cases: Dict[str, Dict[str, Any]] = {}
        self._conn = None
        self._init_db()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    @staticmethod
    def generate_key(model_name: str, payload: Any, params: Dict[str, Any]) -> str:
        if isinstance(payload, (dict, list)):
            stable_payload = json.dumps(payload, sort_keys=True)
        else:
            stable_payload = str(payload)
            
        stable_params = json.dumps(params, sort_keys=True)
        
        raw_string = f"{model_name}::{stable_payload}::{stable_params}"
        return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        # In 'runs_only' mode we skip the SQLite check per original requirements
        if self.mode in ("runs_only", "none"):
            return None

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM global_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, cache_key: str, result_payload: Dict[str, Any]) -> None:
        if self.mode == "none":
            return
        payload_str = json.dumps(result_payload, sort_keys=True)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO global_cache (cache_key, payload)
            VALUES (?, ?)
        """, (cache_key, payload_str))
        conn.commit()

    def preload_legacy_cache(self, model_name: str, agent_mode: str, base_class: str, ignore_errors: bool = False) -> None:
        """Preloads legacy run cache into memory for fast sample_id lookups."""
        if self.mode == "cache_file":
            return
            
        if self.base_runs_dir.exists():
            self._legacy_cases = load_legacy_runs(self.base_runs_dir, model_name, agent_mode, base_class, ignore_errors)
        self._legacy_cache_loaded = True

    def lookup(self, cache_key: str, sample_id: str, model_name: str, agent_mode: str, base_class: str, ignore_errors: bool = False, input_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Multi-tier check:
          1. Check SQLite DB (cache_file)
          2. Check legacy runs via sample_id (and optionally text match)
          3. If found in legacy, migrate to SQLite DB
        """
        if self.mode == "none":
            return None

        # Tier 1: Check SQLite
        if self.mode != "runs_only":
            res = self.get(cache_key)
            if res is not None:
                if ignore_errors and (str(res.get("prediction", "")).strip().lower() == "error" or res.get("error")):
                    res = None
                else:
                    res["from_cache"] = True
                    return res
            
        # Tier 2: Check legacy runs
        if self.mode != "cache_file":
            if not self._legacy_cache_loaded:
                self.preload_legacy_cache(model_name, agent_mode, base_class, ignore_errors)
                
            legacy_res = self._legacy_cases.get(str(sample_id))
            if legacy_res:
                if input_text is not None and legacy_res.get("text", "").strip() != input_text.strip():
                    legacy_res = None
                elif ignore_errors and (str(legacy_res.get("prediction", "")).strip().lower() == "error" or legacy_res.get("error")):
                    legacy_res = None
                else:
                    legacy_res["from_cache"] = True
                    return legacy_res
                
        return None
