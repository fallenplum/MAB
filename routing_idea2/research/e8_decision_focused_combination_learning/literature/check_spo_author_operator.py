"""Execute unchanged PyEPO SPOPlusFunc on the exact 20-action oracle.

This is a source-operator equivalence check on synthetic tensors. It is not a
replication of the author's original routing/portfolio/shortest-path results.
"""
import __future__
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from torch.autograd import Function

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("E8_LITERATURE_WORK", str(HERE / "private")))


def _solve_exact(cost, module):
    index = cost.argmin(1) if module.optmodel.modelSense == "min" else cost.argmax(1)
    solution = torch.nn.functional.one_hot(index, num_classes=cost.shape[1]).to(cost.dtype)
    return solution, (solution*cost).sum(1)


def check():
    source = ROOT / "upstream/PyEPO/pkg/pyepo/func/surrogate.py"
    raw = source.read_bytes()
    parsed = ast.parse(raw.decode("utf-8"))
    nodes = [x for x in parsed.body if isinstance(x, ast.ClassDef) and x.name == "SPOPlusFunc"]
    env = {"torch": torch, "Function": Function, "is_minimize": lambda sense: sense == "min",
           "_solve_or_cache": _solve_exact, "objective_offset": lambda model: 0.}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag), env)
    operator = env["SPOPlusFunc"]
    rng = np.random.default_rng(2026111401)
    maximum_error = 0.
    checks = []
    for variant in ["lex_true_tie", "uniform_true_tie"]:
        r = torch.tensor(rng.integers(0,4,size=(24,20))/3., dtype=torch.float64)
        r[0] = 0.
        r[1,:] = 1.
        r[2,:] = torch.tensor([1.,1.]+[0.]*18)
        q = torch.tensor(rng.normal(0,.3,size=(24,20)), dtype=torch.float64, requires_grad=True)
        if variant == "lex_true_tie":
            true_w = torch.nn.functional.one_hot(r.argmax(1), num_classes=20).to(torch.float64)
        else:
            true_w = (r == r.max(1,keepdim=True).values).to(torch.float64)
            true_w = true_w / true_w.sum(1,keepdim=True)
        true_obj = r.max(1).values
        module = SimpleNamespace(optmodel=SimpleNamespace(modelSense="max"))
        native = operator.apply(q, r, true_w, true_obj, module)
        native.sum().backward()
        expected = (2*q.detach()-r).max(1).values - 2*(q.detach()*true_w).sum(1)+true_obj
        native_gradient = q.grad.detach().clone()
        max_idx = (2*q.detach()-r).argmax(1)
        expected_gradient = 2*(torch.nn.functional.one_hot(max_idx,20).to(torch.float64)-true_w)
        loss_error = float(abs(native.detach()-expected).max())
        gradient_error = float(abs(native_gradient-expected_gradient).max())
        shift = torch.tensor(rng.normal(size=(24,1)), dtype=torch.float64)
        shifted = operator.apply(q.detach()+shift, r, true_w, true_obj, module)
        shift_error = float(abs(native.detach()-shifted).max())
        selected = q.detach().argmax(1)
        regret = true_obj-r.gather(1,selected[:,None]).squeeze(1)
        assert torch.all(native.detach() >= regret-1e-12)
        assert min(loss_error, gradient_error, shift_error) >= 0
        assert max(loss_error, gradient_error, shift_error) < 1e-12
        maximum_error = max(maximum_error,loss_error,gradient_error,shift_error)
        checks.append({"variant":variant,"batch":24,"actions":20,"native_forward_error":loss_error,
                       "native_backward_error":gradient_error,"common_offset_invariance_error":shift_error,
                       "minimum_surrogate":float(native.min()),"minimum_upper_bound_slack":float((native-regret).min())})
    result = {"status":"PASS","kind":"executed exact author operator on synthetic finite-action inputs",
              "source_commit":"e5d768e2749bccdeba7494b4ec848c3eb3ffbd25","source_path":"pkg/pyepo/func/surrogate.py",
              "source_sha256":hashlib.sha256(raw).hexdigest(),"cases":checks,"maximum_error":maximum_error,
              "physical_LLM_calls":0,"note":"true and perturbed ties use frozen first-index enumeration; true-tie averaging is separately tested"}
    output = ROOT / "E8_AUTHOR_SPO_OPERATOR_CHECK.json"
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(check(),indent=2))
