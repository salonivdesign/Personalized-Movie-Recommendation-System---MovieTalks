"""Compare required CF algorithms on a per-user chronological holdout."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.college_data import load_college_dataset
from recsys.neighbors import UserBasedCF, ItemBasedCF
from recsys.svd_model import SVDRecommender
ROOT=Path(__file__).resolve().parents[1]
df, cleaning=load_college_dataset(ROOT/'data/raw/dataset.csv')
# Keep only users with >=3 distinct ratings for holdout; each user's latest rating is test.
df=df.sort_values(['user_id','timestamp','movie_id']).reset_index(drop=True)
counts=df.groupby('user_id').size()
eligible=set(counts[counts>=3].index)
test=df[df.user_id.isin(eligible)].groupby('user_id', group_keys=False).tail(1)
train=df.drop(test.index).reset_index(drop=True); test=test.reset_index(drop=True)
if train.empty or test.empty: raise SystemExit('Not enough data for a train/test evaluation.')
models={'User-based CF':UserBasedCF(),'Item-based CF':ItemBasedCF(),'SVD':SVDRecommender(n_factors=3)}
results=[]
for name, model in models.items():
    model.fit(train)
    pred=model.predict_pairs(test.user_id.to_numpy(), test.movie_id.to_numpy())
    rmse=float(np.sqrt(mean_squared_error(test.rating.to_numpy(), pred)))
    # Top-K relevance: held-out ratings >=4; candidates are items not in that user's training history.
    hits=total=0; per_user_precision=[]; k=3
    train_seen=train.groupby('user_id').movie_id.apply(set).to_dict()
    truth=test[test.rating>=4].groupby('user_id').movie_id.apply(set).to_dict()
    for uid in test.user_id.unique():
        seen=train_seen.get(uid,set()); candidates=[int(i) for i in model.items_ if int(i) not in seen] if hasattr(model,'items_') else []
        relevant=truth.get(uid,set())
        if not candidates or not relevant: continue
        scores=[(int(i),float(model.predict(int(uid),int(i)))) for i in candidates]
        ranked=[i for i,_ in sorted(scores,key=lambda x:x[1],reverse=True)[:min(k,len(scores))]]
        denom=min(k,len(ranked)); p=sum(i in relevant for i in ranked)/denom if denom else 0
        per_user_precision.append(p); hits += sum(i in relevant for i in ranked); total += denom
    precision=float(np.mean(per_user_precision)) if per_user_precision else 0.0
    results.append({'model':name,'RMSE':round(rmse,4),'Precision@3':round(precision,4),'test_ratings':int(len(test)),'users_with_relevant_test_item':len(per_user_precision)})
report={'dataset':'college-provided dataset.csv','cleaning':cleaning,'split':'For users with at least 3 ratings, hold out each eligible user\'s latest rating; other rows are training data.','relevance_threshold':'rating >= 4.0','results':results,'limitations':['Only a small number of distinct movie titles are present; ranking metrics are unstable and Precision@3 is only illustrative.','The supplied genre/language/release-year metadata is inconsistent by title.','This is a small holdout, not evidence of production-quality recommendation performance.']}
out=ROOT/'docs/evaluation_results.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
pd.DataFrame(results).to_csv(ROOT/'docs/evaluation_results.csv',index=False)
print(json.dumps(report,indent=2))
