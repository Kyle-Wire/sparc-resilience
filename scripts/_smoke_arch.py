"""Smoke test for the 6 new architectural modules."""
import sys

passed = []
failed = []

def check(label, fn):
    try:
        fn()
        passed.append(label)
        print(f"  OK  {label}")
    except Exception as e:
        failed.append(label)
        print(f"FAIL  {label}: {e}")

check("ProjectConfig", lambda: __import__("sparc.config.project_config", fromlist=["ProjectConfig"]))
check("ModelSpec", lambda: __import__("sparc.models.spec", fromlist=["ModelSpec"]))
check("PhysicsConstraint", lambda: __import__("sparc.physics.constraint", fromlist=["PhysicsConstraint"]))
check("CausalInputs", lambda: __import__("sparc.interventions.causal_inputs", fromlist=["CausalInputs"]))
check("SpatialCVEngine", lambda: __import__("sparc.run.cv_engine", fromlist=["SpatialCVEngine"]))
check("run.orchestration", lambda: __import__("sparc.run.orchestration", fromlist=["RunContext"]))
check("run.cv", lambda: __import__("sparc.run.cv", fromlist=["SpatialCVEngine"]))
check("run.stages", lambda: __import__("sparc.run.stages", fromlist=["run_stage0"]))
check("run.training", lambda: __import__("sparc.run.training", fromlist=["run_neural"]))

print()
print(f"{len(passed)} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
