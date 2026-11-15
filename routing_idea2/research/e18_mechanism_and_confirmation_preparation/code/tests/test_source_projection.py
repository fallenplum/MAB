"""Known-answer controls for input-only parsing and duplicate screening."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_sources import INPUT_FIELDS, _private_signature, _tokens, canonical_hash, official_example_count, project
from candidate_proposal import protected_screen


class SourceProjectionTests(unittest.TestCase):
    def test_unknown_reference_value_is_syntax_skipped_without_json_decoding(self):
        raw=b'{"question_id":"fixture","question":"What is2+2?","reference":UNAUTHORIZED_VALUE_MUST_NOT_BE_JSON_DECODED,"member_outputs":{"score":UNAUTHORIZED_SCORE},"task":"gsm8k","subtask":null,"source_split":"train"}'
        row=project(raw,INPUT_FIELDS)
        self.assertEqual(set(row),INPUT_FIELDS)
        self.assertNotIn("reference",row)
        self.assertNotIn("member_outputs",row)

    def test_metadata_projection_does_not_decode_question_value(self):
        row=project(b'{"question_id":"fixture","question":DO_NOT_JSON_DECODE,"source_index":7,"reference":DO_NOT_JSON_DECODE}',{"question_id","source_index"})
        self.assertEqual(row,{"question_id":"fixture","source_index":7})

    def test_duplicate_object_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            project(b'{"question_id":"a","question_id":"b"}',{"question_id"})

    def test_nfkc_and_whitespace_canonical_rule(self):
        self.assertEqual(canonical_hash("Ａ   has\n2 apples."),canonical_hash("A has 2 apples."))

    def test_numbers_entities_and_operators_remain_distinct_instances(self):
        self.assertNotEqual(_tokens("Alice has2 apples."),_tokens("Alice has3 apples."))
        self.assertNotEqual(_tokens("Alice has2 apples."),_tokens("Bob has2 apples."))
        self.assertNotEqual(_tokens("2+3"),_tokens("2-3"))

    def test_official_counts_do_not_decode_example_input_or_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic.json"
            path.write_text('{"examples":[{"input":DO_NOT_DECODE,"target":DO_NOT_DECODE},{"input":DO_NOT_DECODE,"target":DO_NOT_DECODE}]}',encoding="utf-8")
            self.assertEqual(official_example_count(path),2)

    def test_private_signature_does_not_persist_plaintext(self):
        row={"question":"Alice has2 apples.","task":"gsm8k","subtask":None}
        signature=_private_signature(row,bytes(range(32)))
        self.assertNotIn("Alice",str(signature))
        self.assertNotIn("apples",str(signature))
        self.assertEqual(signature,_private_signature(row,bytes(range(32))))

    def test_protected_format_match_excludes_and_different_instances_survive(self):
        key=bytes(range(32))
        protected={"question_id":"protected",**_private_signature({"question":"Alice has2 apples.","task":"gsm8k","subtask":None},key)}
        rows=[dict(question_id=str(index),task="gsm8k",subtask=None,registered_excluded=False,question=text) for index,text in enumerate(("Alice has2 apples!","Alice has3 apples.","Bob has2 apples."))]
        removed,edges,summary=protected_screen(rows,key,[protected])
        self.assertEqual(removed,{0})
        self.assertEqual(len(edges),1)
        self.assertEqual(summary["additional_protected_input_values_decoded"],0)


if __name__=="__main__":
    unittest.main()
