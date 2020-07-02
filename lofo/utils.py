import multiprocessing
import pandas as pd


def lofo_to_df(lofo_scores, feature_list):
    importance_df = pd.DataFrame()
    importance_df["feature"] = feature_list
    importance_df["importance_mean"] = lofo_scores.mean(axis=1)
    importance_df["importance_std"] = lofo_scores.std(axis=1)

    for val_score in range(lofo_scores.shape[1]):
        importance_df["val_imp_{}".format(val_score)] = lofo_scores[:, val_score]

    return importance_df.sort_values("importance_mean", ascending=False)


def parallel_apply(cv_func, feature_list, n_jobs):
    pool = multiprocessing.Pool(n_jobs)
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for f in feature_list:
        pool.apply_async(cv_func, (f, result_queue))

    pool.close()
    pool.join()

    lofo_cv_result = [result_queue.get() for _ in range(len(feature_list))]
    return lofo_cv_result
