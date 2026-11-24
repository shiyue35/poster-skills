"""Optional in-process boundary for a trusted host; no image provider is connected.

The host owns policies, authenticated review results, keys, one shared BudgetLedger
and the sender. This module cannot intercept the desktop's built-in image_gen tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import threading
from typing import Callable

REVIEW_COUNTS = {
    "composition": 5,
    "generate": 10,
    "edit": 10,
    "component": 10,
    "extend": 10,
}


class Rejected(RuntimeError):
    """This rejected operation did not reach its configured sender."""


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def absolute(path: str) -> str:
    return str(Path(path).resolve(strict=True))


@dataclass(frozen=True)
class Policy:
    # Only the trusted host may compile or replace this configuration.
    task_id: str
    request_id: str
    subject: str
    direction: str
    style: str
    version: str
    stage: str
    tool: str
    producer_id: str
    authorization_id: str
    is_revision: bool
    native_size: tuple[int, int]
    exact_text: tuple[str, ...]
    references: tuple[str, ...]
    sources: tuple[str, ...]
    max_calls: int = 3
    max_revisions: int = 2


def validate_policy(policy: Policy) -> None:
    if policy.stage not in REVIEW_COUNTS:
        raise Rejected("unknown stage")
    for key in ("task_id", "request_id", "subject", "direction", "style", "version",
                "tool", "producer_id", "authorization_id"):
        value = getattr(policy, key)
        if not isinstance(value, str) or not value.strip():
            raise Rejected(f"missing trusted policy field: {key}")
    if type(policy.is_revision) is not bool:
        raise Rejected("invalid revision classification")
    if policy.stage == "edit" and not policy.is_revision:
        raise Rejected("edit must be classified as a revision")
    if (len(policy.native_size) != 2
            or any(type(v) is not int or v <= 0 for v in policy.native_size)):
        raise Rejected("invalid native target")
    if any(type(v) is not int or v < 0
           for v in (policy.max_calls, policy.max_revisions)):
        raise Rejected("invalid trusted budget")


@dataclass(frozen=True)
class Snapshot:
    manifest_json: str
    digest: str
    images: tuple[tuple[str, bytes], ...]

    def checked_manifest(self) -> dict:
        try:
            manifest = json.loads(self.manifest_json)
            if sha256(self.manifest_json.encode("utf-8")) != self.digest:
                raise Rejected("snapshot content changed")
            actual = [{"path": path, "sha256": sha256(data)} for path, data in self.images]
            if actual != manifest["references"]:
                raise Rejected("verified image bytes changed")
            return manifest
        except Rejected:
            raise
        except Exception as exc:
            raise Rejected("invalid snapshot") from exc


def prepare(policy: Policy, request: dict) -> Snapshot:
    """Check the host policy and freeze the exact request and image bytes."""
    try:
        validate_policy(policy)
        scope = ("task_id", "request_id", "subject", "direction", "style", "version",
                 "stage", "tool", "is_revision")
        expected = set(scope) | {"native_size", "arguments"}
        if not isinstance(request, dict) or set(request) != expected:
            raise Rejected("request schema mismatch; self-declared approval is not accepted")
        for key in scope:
            if request[key] != getattr(policy, key):
                raise Rejected(f"request scope mismatch: {key}")
        if type(request["is_revision"]) is not bool:
            raise Rejected("invalid request revision flag")
        if list(request["native_size"]) != list(policy.native_size):
            raise Rejected("native target changed")
        args = request["arguments"]
        required = {"prompt", "transparent_background"}
        if (not isinstance(args, dict) or not required.issubset(args)
                or set(args) - required - {"referenced_image_paths"}):
            raise Rejected("tool argument schema mismatch")
        prompt = args["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise Rejected("missing prompt")
        if type(args["transparent_background"]) is not bool:
            raise Rejected("invalid background flag")
        if any(not isinstance(text, str) or not text or text not in prompt
               for text in policy.exact_text):
            raise Rejected("required verbatim text missing")
        size_tokens = (f"{policy.native_size[0]}×{policy.native_size[1]}",
                       f"{policy.native_size[0]}x{policy.native_size[1]}")
        if not any(token in prompt for token in size_tokens):
            raise Rejected("native target absent from the actual prompt")
        refs = args.get("referenced_image_paths", [])
        if not isinstance(refs, list) or not all(isinstance(p, str) for p in refs):
            raise Rejected("invalid reference list")
        if policy.stage != "composition" and not refs:
            raise Rejected("production requires reference images")
        if [absolute(p) for p in refs] != [absolute(p) for p in policy.references]:
            raise Rejected("required references missing, replaced, reordered or added")
        if not policy.sources:
            raise Rejected("missing original requirement sources")
        sources = [{"path": absolute(p), "sha256": sha256(Path(p).read_bytes())}
                   for p in policy.sources]
        images = tuple((absolute(p), Path(p).read_bytes()) for p in refs)
        manifest = {
            "policy": asdict(policy), "request": request, "sources": sources,
            "references": [{"path": p, "sha256": sha256(data)} for p, data in images],
        }
        serialized = canonical(manifest)
        return Snapshot(serialized, sha256(serialized.encode("utf-8")), images)
    except Rejected:
        raise
    except Exception as exc:
        raise Rejected("preflight could not complete") from exc


class ReviewAuthority:
    """The host authenticates the real reviewer before issuing a receipt.

    A signature alone does not prove a sub-agent ran or that an instruction was
    correctly understood. Drafting agents must not own this object or its key.
    """

    def __init__(self, producer_id: str, reviewer_ids: frozenset[str]):
        self.producer_id = producer_id
        self.reviewer_ids = reviewer_ids
        self.__key = secrets.token_bytes(32)

    def _sign(self, body: dict) -> str:
        return hmac.new(self.__key, canonical(body).encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def issue(self, snapshot: Snapshot, reviewer_id: str, checks: list[dict],
              *, decision: str = "pass", findings: tuple[str, ...] = ()) -> dict:
        snapshot.checked_manifest()
        if reviewer_id == self.producer_id or reviewer_id not in self.reviewer_ids:
            raise Rejected("reviewer is not an authenticated independent reviewer")
        body = {
            "review_id": secrets.token_hex(16), "digest": snapshot.digest,
            "producer_id": self.producer_id, "reviewer_id": reviewer_id,
            "decision": decision, "checks": checks, "findings": list(findings),
        }
        body = json.loads(canonical(body))
        return {"body": body, "signature": self._sign(body)}

    def verify(self, snapshot: Snapshot, receipt: dict) -> str:
        try:
            manifest = snapshot.checked_manifest()
            if not isinstance(receipt, dict) or set(receipt) != {"body", "signature"}:
                raise Rejected("missing authenticated review receipt")
            body = receipt["body"]
            if not hmac.compare_digest(self._sign(body), receipt["signature"]):
                raise Rejected("review signature invalid")
            if body["digest"] != snapshot.digest or body["producer_id"] != self.producer_id:
                raise Rejected("review does not cover this request and these sources")
            if (body["reviewer_id"] == self.producer_id
                    or body["reviewer_id"] not in self.reviewer_ids):
                raise Rejected("independent reviewer mismatch")
            if body["decision"] != "pass" or body["findings"]:
                raise Rejected("review has unresolved findings")
            if not isinstance(body["review_id"], str) or not body["review_id"]:
                raise Rejected("missing review identity")
            count = REVIEW_COUNTS[manifest["policy"]["stage"]]
            checks = body["checks"]
            if (not isinstance(checks, list) or len(checks) != count
                    or any(type(item["id"]) is not int for item in checks)
                    or {item["id"] for item in checks} != set(range(1, count + 1))):
                raise Rejected(f"{count}-item review incomplete for this stage")
            if any(item["status"] != "pass" or not isinstance(item["evidence"], str)
                   or not item["evidence"].strip() for item in checks):
                raise Rejected("review contains missing evidence, gaps or conflicts")
            return body["review_id"]
        except Rejected:
            raise
        except Exception as exc:
            raise Rejected("review verification failed") from exc


class BudgetLedger:
    """Host-owned, shared in-process counts and consumed review identities.

    The host must preserve/restore this state across restarts. Creating a new
    Executor is safe only when it keeps using this same ledger.
    """

    def __init__(self):
        self.__counts: dict[tuple[str, str, str, str], tuple[int, int]] = {}
        self.__used_reviews: set[str] = set()
        self.__lock = threading.Lock()

    @staticmethod
    def _key(policy: Policy) -> tuple[str, str, str, str]:
        return (policy.task_id, policy.request_id, policy.subject, policy.direction)

    def usage(self, policy: Policy) -> dict[str, int]:
        with self.__lock:
            calls, revisions = self.__counts.get(self._key(policy), (0, 0))
            return {"calls": calls, "revisions": revisions}

    def claim(self, policy: Policy, review_id: str) -> None:
        with self.__lock:
            if review_id in self.__used_reviews:
                raise Rejected("review receipt already used")
            key = self._key(policy)
            calls, revisions = self.__counts.get(key, (0, 0))
            if calls >= policy.max_calls:
                raise Rejected("total call budget exhausted")
            if policy.is_revision and revisions >= policy.max_revisions:
                raise Rejected("revision budget exhausted")
            self.__used_reviews.add(review_id)
            self.__counts[key] = (calls + 1, revisions + int(policy.is_revision))


class ProviderBoundary:
    """No default transport; only valid single-use permits reach the sender."""

    def __init__(self, key: bytes, policy: Policy, ledger: BudgetLedger,
                 sender: Callable[[Snapshot], object] | None):
        self.__key = key
        self.__policy = policy
        self.__ledger = ledger
        self.__sender = sender

    def dispatch(self, snapshot: Snapshot, permit: dict | None) -> object:
        try:
            if not isinstance(permit, dict) or set(permit) != {"body", "signature"}:
                raise Rejected("provider requires a verified execution permit")
            body = permit["body"]
            if set(body) != {"purpose", "digest", "nonce", "review_id"}:
                raise Rejected("execution permit schema mismatch")
            expected = hmac.new(self.__key, canonical(body).encode("utf-8"),
                               hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, permit["signature"]):
                raise Rejected("execution permit invalid")
            if body["digest"] != snapshot.digest or body["purpose"] != "execute":
                raise Rejected("execution permit does not cover this input")
            manifest = snapshot.checked_manifest()
            if not isinstance(body["nonce"], str) or not body["nonce"]:
                raise Rejected("execution permit missing identity")
            if not isinstance(body["review_id"], str) or not body["review_id"]:
                raise Rejected("execution permit missing review")
            if manifest["policy"] != json.loads(canonical(asdict(self.__policy))):
                raise Rejected("provider policy mismatch")
            for source in manifest["sources"]:
                if sha256(Path(source["path"]).read_bytes()) != source["sha256"]:
                    raise Rejected("requirement source changed after authorization")
            if self.__sender is None:
                raise Rejected("no provider connected; live generation is disabled")
            self.__ledger.claim(self.__policy, body["review_id"])
        except Rejected:
            raise
        except Exception as exc:
            raise Rejected("provider boundary verification failed") from exc
        # After this point the sender may have side effects. Failures consume the
        # call (and revision, when applicable); never automatically retry/refund.
        return self.__sender(snapshot)


class Executor:
    def __init__(self, policy: Policy, reviews: ReviewAuthority, ledger: BudgetLedger,
                 sender: Callable[[Snapshot], object] | None = None):
        validate_policy(policy)
        if policy.producer_id != reviews.producer_id:
            raise Rejected("trusted producer configuration mismatch")
        self.policy = policy
        self.reviews = reviews
        self.__key = secrets.token_bytes(32)
        self.provider = ProviderBoundary(self.__key, policy, ledger, sender)

    def authorize(self, request: dict, receipt: dict) -> tuple[Snapshot, dict]:
        try:
            snapshot = prepare(self.policy, request)
            review_id = self.reviews.verify(snapshot, receipt)
        except Rejected:
            raise
        except Exception as exc:
            raise Rejected("executor preflight failed; provider not called") from exc
        body = {
            "purpose": "execute", "digest": snapshot.digest,
            "nonce": secrets.token_hex(16), "review_id": review_id,
        }
        signature = hmac.new(self.__key, canonical(body).encode("utf-8"),
                             hashlib.sha256).hexdigest()
        return snapshot, {"body": body, "signature": signature}

    def execute(self, request: dict, receipt: dict) -> object:
        snapshot, permit = self.authorize(request, receipt)
        return self.provider.dispatch(snapshot, permit)
