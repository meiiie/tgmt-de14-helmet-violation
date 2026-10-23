"""Linear probe / kNN trên embedding DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m, gated) cho crop đầu.
Đánh giá GroupKFold theo clip (không rò rỉ clip), so sánh với YOLO helmet (nnsohamnn) chạy trên cùng crop. Lưu probe cuối."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from de14.classifiers import DinoEmbedder, DinoProbe, YoloHelmetCls
mid = sys.argv[1] if len(sys.argv) > 1 else "facebook/dinov3-vits16-pretrain-lvd1689m"
d = np.load("data/crops/train_heads.npz"); X, Y = d["X"], d["Y"]; G = np.array([m.split("/")[0] for m in d["meta"]])
emb = DinoEmbedder(mid); t = time.time(); F = emb(list(X)); te = time.time() - t
print("model", emb.model_id, "crops", len(Y), "pos", int(Y.sum()), f"embed {len(Y)/te:.1f} crops/s")
def m(y, p, s): return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p), "f1": f1_score(y, p), "auc": roc_auc_score(y, s)}
res = {}
for name, mk in [("dino_linear", lambda: LogisticRegression(max_iter=3000, class_weight="balanced")),
                 ("dino_knn7", lambda: KNeighborsClassifier(7, weights="distance"))]:
    P = np.zeros(len(Y)); S = np.zeros(len(Y))
    for tr, te_ in GroupKFold(5).split(F, Y, G):
        c = mk().fit(F[tr], Y[tr]); S[te_] = c.predict_proba(F[te_])[:, 1]
    res[name] = m(Y, (S >= 0.5).astype(int), S)
yc = YoloHelmetCls("weights/helmet_yolo11s_nnsohamnn.pt")
ps = yc(list(X)); cov = np.array([p is not None for p in ps]); s = np.array([p if p is not None else 0.5 for p in ps])
r = m(Y[cov], (s[cov] >= 0.5).astype(int), s[cov]); r["coverage"] = float(cov.mean()); res["yolo_helmet_on_crop"] = r
print(json.dumps(res, indent=1))
os.makedirs("outputs/metrics", exist_ok=True)
json.dump({"model": emb.model_id, "n": int(len(Y)), "pos": int(Y.sum()), "clips": int(len(set(G))), **res},
          open(f"outputs/metrics/probe_cv_{emb.model_id.split('/')[-1]}.json", "w"), indent=2)
p = DinoProbe(emb, "linear"); p.clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(F, Y)
p.save("weights/dino_probe.joblib"); print("saved weights/dino_probe.joblib")
