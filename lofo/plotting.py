def plot_importance(importance_df, figsize=(8, 8)):
    """Plot feature importance

    Parameters
    ----------
    importance_df : pandas dataframe
        Output dataframe from LOFO/FLOFO get_importance
    figsize : tuple
    """
    importance_df = importance_df.copy()
    importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
    importance_df.sort_values("importance_mean", inplace=True)

    importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                       kind='barh', color=importance_df["color"], figsize=figsize)
