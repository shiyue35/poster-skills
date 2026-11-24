"""Reproducible local tests. No real sub-agent, network or image service is used.

Run: python -B -X utf8 scripts/test_preflight_gate.py
Temporary fixtures are created only under the OS temp directory.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
import unittest

sys.dont_write_bytecode = True
from preflight_gate import (BudgetLedger, Executor, Policy, Rejected,
                            REVIEW_COUNTS, ReviewAuthority, prepare)

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opdt-test-")
        self.root = Path(self.temp.name).resolve()
        self.allowed_parent = Path(tempfile.gettempdir()).resolve()
        self.assertEqual(self.root.parent, self.allowed_parent)
        self.source = self.root / "brief.md"
        self.source.write_text("Local mock authorization and scope evidence.", encoding="utf-8")
        self.image = self.root / "composition.png"
        self.image.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j4f8AAAAASUVORK5CYII="))
        self.calls = []
        self.ledger = BudgetLedger()
        self.reviews = ReviewAuthority("maker", frozenset({"reviewer"}))
        self.policy = self.make_policy()

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        if (target != self.root or target.parent != self.allowed_parent
                or not target.name.startswith("opdt-test-")):
            raise RuntimeError("refusing cleanup outside this test's temporary directory")
        self.temp.cleanup()

    def make_policy(self, **changes):
        base = Policy(
            task_id="task", request_id="user-request-1", subject="subject",
            direction="chosen-A", style="style", version="v001",
            stage="composition", tool="mock-image", producer_id="maker",
            authorization_id="mock-user-decision", is_revision=False,
            native_size=(1080, 1920), exact_text=("Title", "Subject"),
            references=(), sources=(str(self.source),))
        return replace(base, **changes)

    def request(self, policy=None):
        policy = policy or self.policy
        keys = ("task_id", "request_id", "subject", "direction", "style",
                "version", "stage", "tool", "is_revision")
        request = {key: getattr(policy, key) for key in keys}
        request["native_size"] = list(policy.native_size)
        request["arguments"] = {
            "prompt": f"Title Subject {policy.native_size[0]}x{policy.native_size[1]}",
            "transparent_background": False,
        }
        if policy.references:
            request["arguments"]["referenced_image_paths"] = list(policy.references)
        return request

    def receipt(self, policy=None, request=None, count=None, checks=None):
        policy = policy or self.policy
        request = request or self.request(policy)
        count = (5 if policy.stage == "composition" else 10) if count is None else count
        checks = checks if checks is not None else [
            {"id": i, "status": "pass", "evidence": "Local mock raw evidence"}
            for i in range(1, count + 1)]
        return self.reviews.issue(prepare(policy, request), "reviewer", checks)

    def send(self, snapshot):
        self.calls.append(snapshot)
        return {"mock": True}

    def executor(self, policy=None, sender=None):
        return Executor(policy or self.policy, self.reviews, self.ledger,
                        self.send if sender is None else sender)

    def execute_fresh(self, policy=None):
        policy = policy or self.policy
        request = self.request(policy)
        return self.executor(policy).execute(request, self.receipt(policy, request))

    def blocked(self, callback):
        before = len(self.calls)
        with self.assertRaises(Rejected):
            callback()
        self.assertEqual(len(self.calls), before)

    def production_policy(self, stage="generate", **changes):
        return self.make_policy(stage=stage, references=(str(self.image),),
                                is_revision=(stage == "edit"), **changes)

    def test_all_supported_stages_reach_mock_sender(self):
        for stage in ("composition", "generate", "edit", "component", "extend"):
            with self.subTest(stage=stage):
                policy = (self.make_policy(direction=stage) if stage == "composition"
                          else self.production_policy(stage, direction=stage))
                self.execute_fresh(policy)
        self.assertEqual(len(self.calls), 5)

    def test_composition_needs_no_existing_image_or_reference_argument(self):
        self.execute_fresh()
        self.assertEqual(self.calls[0].images, ())

    def test_composition_accepts_explicit_empty_references(self):
        request = self.request()
        request["arguments"]["referenced_image_paths"] = []
        self.executor().execute(request, self.receipt(request=request))
        self.assertEqual(len(self.calls), 1)

    def test_composition_can_edit_an_existing_diagram(self):
        policy = self.make_policy(references=(str(self.image),), is_revision=True)
        self.execute_fresh(policy)
        self.assertEqual(self.ledger.usage(policy), {"calls": 1, "revisions": 1})

    def test_production_without_reference_rejected(self):
        for stage in ("generate", "edit", "component", "extend"):
            with self.subTest(stage=stage):
                policy = self.make_policy(stage=stage, is_revision=(stage == "edit"))
                self.blocked(lambda: prepare(policy, self.request(policy)))

    def test_unknown_stage_rejected(self):
        for stage in ("compostion", "export", "", "unexpected"):
            with self.subTest(stage=stage):
                self.blocked(lambda: self.executor(self.make_policy(stage=stage)))

    def test_missing_trusted_request_or_direction_rejected(self):
        for changes in ({"request_id": ""}, {"direction": ""}, {"authorization_id": ""}):
            with self.subTest(changes=changes):
                self.blocked(lambda: self.executor(self.make_policy(**changes)))

    def test_edit_requires_revision_classification(self):
        policy = self.make_policy(stage="edit", references=(str(self.image),))
        self.blocked(lambda: self.executor(policy))

    def test_wrong_review_count_cannot_cross_stage(self):
        for policy, count in ((self.policy, 10), (self.production_policy(), 5)):
            with self.subTest(stage=policy.stage):
                request = self.request(policy)
                receipt = self.receipt(policy, request, count=count)
                self.blocked(lambda: self.executor(policy).execute(request, receipt))

    def test_producer_cannot_downgrade_stage(self):
        policy = self.production_policy()
        request = self.request(policy)
        receipt = self.receipt(policy, request)
        request["stage"] = "composition"
        self.blocked(lambda: self.executor(policy).execute(request, receipt))

    def test_producer_cannot_change_budget_identity(self):
        for key in ("request_id", "direction", "subject"):
            request = self.request()
            receipt = self.receipt(request=request)
            request[key] = "invented"
            self.blocked(lambda: self.executor().execute(request, receipt))

    def test_self_declared_approval_rejected(self):
        request = self.request()
        request["passed"] = True
        self.blocked(lambda: prepare(self.policy, request))

    def test_unknown_tool_argument_rejected(self):
        request = self.request()
        request["arguments"]["skip_review"] = True
        self.blocked(lambda: prepare(self.policy, request))

    def test_target_and_verbatim_text_are_checked(self):
        for kind in ("size", "text"):
            request = self.request()
            if kind == "size":
                request["native_size"] = [1920, 1080]
            else:
                request["arguments"]["prompt"] = "Title 1080x1920"
            self.blocked(lambda: prepare(self.policy, request))

    def test_missing_and_reordered_reference_rejected(self):
        other = self.root / "other.png"
        other.write_bytes(self.image.read_bytes() + b"other")
        policy = self.make_policy(references=(str(self.image), str(other)))
        request = self.request(policy)
        receipt = self.receipt(policy, request)
        request["arguments"]["referenced_image_paths"].reverse()
        self.blocked(lambda: self.executor(policy).execute(request, receipt))
        del request["arguments"]["referenced_image_paths"]
        self.blocked(lambda: self.executor(policy).execute(request, receipt))

    def test_prompt_change_invalidates_review(self):
        request = self.request()
        receipt = self.receipt(request=request)
        request["arguments"]["prompt"] += " changed"
        self.blocked(lambda: self.executor().execute(request, receipt))

    def test_source_change_before_authorize_invalidates_review(self):
        request = self.request()
        receipt = self.receipt(request=request)
        self.source.write_text("changed", encoding="utf-8")
        self.blocked(lambda: self.executor().execute(request, receipt))

    def test_source_change_after_authorize_is_checked_before_send(self):
        request = self.request()
        executor = self.executor()
        snapshot, permit = executor.authorize(request, self.receipt(request=request))
        self.source.write_text("changed", encoding="utf-8")
        self.blocked(lambda: executor.provider.dispatch(snapshot, permit))
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 0)

    def test_image_change_before_authorize_invalidates_review(self):
        policy = self.production_policy()
        request = self.request(policy)
        receipt = self.receipt(policy, request)
        self.image.write_bytes(self.image.read_bytes() + b"changed")
        self.blocked(lambda: self.executor(policy).execute(request, receipt))

    def test_sender_uses_verified_image_bytes_not_later_path_contents(self):
        policy = self.production_policy()
        request = self.request(policy)
        original = self.image.read_bytes()
        executor = self.executor(policy)
        snapshot, permit = executor.authorize(request, self.receipt(policy, request))
        self.image.write_bytes(b"later path contents")
        executor.provider.dispatch(snapshot, permit)
        self.assertEqual(self.calls[0].images[0][1], original)

    def test_forged_signature_rejected(self):
        request = self.request()
        receipt = self.receipt(request=request)
        receipt["signature"] = "0" * 64
        self.blocked(lambda: self.executor().execute(request, receipt))

    def test_same_producer_cannot_issue_independent_review(self):
        snapshot = prepare(self.policy, self.request())
        self.blocked(lambda: self.reviews.issue(snapshot, "maker", []))

    def test_gaps_or_empty_evidence_cannot_pass(self):
        for bad in ("status", "evidence", "id"):
            checks = [{"id": i, "status": "pass", "evidence": "mock raw evidence"}
                      for i in range(1, 6)]
            checks[0][bad] = {"status": "gap", "evidence": "", "id": True}[bad]
            request = self.request()
            receipt = self.receipt(request=request, checks=checks)
            self.blocked(lambda: self.executor().execute(request, receipt))

    def test_no_provider_does_not_charge_or_consume_review(self):
        request = self.request()
        receipt = self.receipt(request=request)
        executor = Executor(self.policy, self.reviews, self.ledger)
        self.blocked(lambda: executor.execute(request, receipt))
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 0)
        self.executor().execute(request, receipt)
        self.assertEqual(len(self.calls), 1)

    def test_direct_dispatch_without_permit_rejected(self):
        executor = self.executor()
        self.blocked(lambda: executor.provider.dispatch(prepare(self.policy, self.request()), None))

    def test_execution_permit_is_single_use(self):
        executor = self.executor()
        request = self.request()
        snapshot, permit = executor.authorize(request, self.receipt(request=request))
        executor.provider.dispatch(snapshot, permit)
        self.blocked(lambda: executor.provider.dispatch(snapshot, permit))

    def test_new_permits_cannot_reuse_same_review_receipt(self):
        executor = self.executor()
        request = self.request()
        receipt = self.receipt(request=request)
        first = executor.authorize(request, receipt)
        second = executor.authorize(request, receipt)
        executor.provider.dispatch(*first)
        self.blocked(lambda: executor.provider.dispatch(*second))
        self.blocked(lambda: self.executor().execute(request, receipt))
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 1)

    def test_fresh_review_for_same_input_can_use_remaining_budget(self):
        a = self.receipt()
        b = self.receipt()
        self.assertNotEqual(a["body"]["review_id"], b["body"]["review_id"])
        self.executor().execute(self.request(), a)
        self.executor().execute(self.request(), b)
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 2)

    def test_total_budget_is_shared_across_executors(self):
        for _ in range(3):
            self.execute_fresh()
        self.blocked(lambda: self.execute_fresh())
        self.assertEqual(self.ledger.usage(self.policy), {"calls": 3, "revisions": 0})

    def test_revision_budget_is_independent_of_remaining_total(self):
        policy = self.production_policy("edit")
        self.execute_fresh(policy)
        self.execute_fresh(policy)
        self.blocked(lambda: self.execute_fresh(policy))
        self.assertEqual(self.ledger.usage(policy), {"calls": 2, "revisions": 2})

    def test_stage_and_version_changes_do_not_reset_budget(self):
        self.execute_fresh()
        self.execute_fresh(self.production_policy(version="v002"))
        self.execute_fresh(self.production_policy("component", version="v003"))
        self.blocked(lambda: self.execute_fresh(self.production_policy("edit", version="v004")))
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 3)

    def test_new_user_request_has_new_budget(self):
        for _ in range(3):
            self.execute_fresh()
        newer = replace(self.policy, request_id="user-request-2")
        self.execute_fresh(newer)
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 3)
        self.assertEqual(self.ledger.usage(newer)["calls"], 1)

    def test_authorized_directions_and_subjects_have_separate_counts(self):
        for _ in range(3):
            self.execute_fresh()
        for policy in (replace(self.policy, direction="chosen-B"),
                       replace(self.policy, subject="another-subject")):
            self.execute_fresh(policy)
            self.assertEqual(self.ledger.usage(policy)["calls"], 1)

    def test_sender_failure_consumes_call_and_revision(self):
        policy = self.production_policy("edit", max_calls=1)
        def failing(snapshot):
            self.calls.append(snapshot)
            raise RuntimeError("mock sender failure")
        request = self.request(policy)
        with self.assertRaisesRegex(RuntimeError, "mock sender failure"):
            self.executor(policy, failing).execute(request, self.receipt(policy, request))
        self.assertEqual(self.ledger.usage(policy), {"calls": 1, "revisions": 1})
        self.blocked(lambda: self.execute_fresh(policy))

    def test_preflight_rejection_does_not_consume_review_or_budget(self):
        request = self.request()
        receipt = self.receipt(request=request)
        request["arguments"]["prompt"] += " unreviewed change"
        self.blocked(lambda: self.executor().execute(request, receipt))
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 0)
        request = self.request()
        self.executor().execute(request, receipt)

    def test_explicit_host_limits_override_defaults(self):
        policy = replace(self.policy, max_calls=5, max_revisions=0)
        for _ in range(5):
            self.execute_fresh(policy)
        self.blocked(lambda: self.execute_fresh(policy))
        self.assertEqual(self.ledger.usage(policy)["calls"], 5)

    def test_zero_budget_and_invalid_budget_are_not_permissive(self):
        self.blocked(lambda: self.execute_fresh(replace(self.policy, max_calls=0)))
        for value in (-1, True, "3"):
            with self.subTest(value=value):
                self.blocked(lambda: self.executor(replace(self.policy, max_calls=value)))
        self.assertEqual(self.calls, [])

    def test_budget_policy_change_requires_a_new_review(self):
        request = self.request()
        receipt = self.receipt(request=request)
        changed = replace(self.policy, max_calls=5)
        self.blocked(lambda: self.executor(changed).execute(self.request(changed), receipt))

    def test_shared_ledger_prevents_parallel_overspend(self):
        def attempt(_):
            try:
                self.execute_fresh()
                return True
            except Rejected:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(sum(results), 3)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.ledger.usage(self.policy)["calls"], 3)

    def test_explicit_shared_ledger_is_required(self):
        with self.assertRaises(TypeError):
            Executor(self.policy, self.reviews)


class PackageTests(unittest.TestCase):
    def test_versions_and_invocation_metadata(self):
        entry = (ROOT / "SKILL.md").read_text(encoding="utf-8-sig")
        version = re.search(r'^\s+version: "([^"]+)"', entry, re.M).group(1)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        for relative in ("references/method.md", "references/brief-template.md",
                         "references/preflight-review.md"):
            self.assertIn("版本 " + version,
                          (ROOT / relative).read_text(encoding="utf-8-sig"))
        ui = (ROOT / "agents/openai.yaml").read_text(encoding="utf-8-sig")
        name = re.search(r"^name: ([a-z0-9-]+)$", entry, re.M).group(1)
        self.assertEqual(ROOT.name, name)
        self.assertIn("$" + name, ui)
        short = re.search(r'short_description: "([^"]+)"', ui).group(1)
        self.assertTrue(25 <= len(short) <= 64)

    def test_documented_program_stages_match_supported_interface(self):
        protocol = (ROOT / "references/preflight-review.md").read_text(encoding="utf-8-sig")
        tick = chr(96)
        declared = set(re.findall(r"^\| " + tick + r"([a-z_]+)" + tick + r" \|",
                                  protocol, re.M))
        self.assertEqual(declared, set(REVIEW_COUNTS))

    def test_internal_links_and_anchors_resolve(self):
        checked = 0
        def slug(title):
            title = title.strip().lower().replace(" ", "-")
            return "".join(c for c in title
                           if c in "_-" or not unicodedata.category(c).startswith(("P", "S")))
        for path in (ROOT / "SKILL.md", *(ROOT / "references").glob("*.md")):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
                if "://" in target:
                    continue
                part, _, anchor = target.partition("#")
                linked = (path.parent / part).resolve() if part else path
                self.assertTrue(linked.is_file(), f"{path}: {target}")
                if anchor and linked.suffix == ".md":
                    headings = re.findall(r"^#{1,6}\s+(.+?)\s*$",
                                          linked.read_text(encoding="utf-8-sig"), re.M)
                    self.assertIn(anchor, {slug(h) for h in headings}, f"{path}: {target}")
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
