import sys, os, importlib, json, time
ds = sys.argv[1]
os.chdir("/Users/andrewmorgan/Dev/gamakon/hff/notebooks/_ledgers/srbench_repo/experiment")
sys.path.insert(0, os.getcwd())
from evaluate_model import evaluate_model
alg = importlib.__import__('methods.hff_sr.regressor', globals(), locals(), ['*'])
alg.est._verbose_fit = True
out = f"/Users/andrewmorgan/Dev/gamakon/hff/notebooks/sr_logs/srbench_debug/results_{ds}"
path = f"/Users/andrewmorgan/Dev/gamakon/hff/notebooks/_ledgers/pmlb_repo/datasets/{ds}/{ds}.tsv.gz"
t = time.time()
jf = evaluate_model(dataset=path, results_path=out, random_state=23654, est_name="hff_sr", est=alg.est, model=alg.model,
                    algorithm=alg, test=False, target_noise=0.0, feature_noise=0.0, **alg.eval_kwargs)
r = json.load(open(jf))
e = alg.est._engine
print(f"RESULT {ds} | model {r['symbolic_model'][:80]} | r2_test {r['r2_test']} | wall {time.time()-t:.1f}s | "
      f"source {getattr(e,'discovered_source_',None)} | generations {getattr(e,'generations_run_',None)}")
