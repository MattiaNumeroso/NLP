"""Modulo per analisi stilometrica: PCA, clustering, classificazione."""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

from .config import FEATURE_GROUPS, MODEL_COLORS


def get_feature_columns(df: pd.DataFrame, groups: list = None) -> list:
    """Restituisce le colonne per i gruppi di features specificati."""
    if groups is None:
        groups = list(FEATURE_GROUPS.keys())

    prefixes = [FEATURE_GROUPS[g] for g in groups if g in FEATURE_GROUPS]

    cols = []
    for col in df.columns:
        for prefix in prefixes:
            if prefix in col:
                cols.append(col)
                break

    return cols   # stesso ordine delle colonne del dataframe (un set cambierebbe ordine a ogni avvio)


def prepare_features(df: pd.DataFrame, groups: list = None) -> tuple:
    """Prepara le features: selezione, fillna, scaling."""
    cols = get_feature_columns(df, groups)
    X = df[cols].fillna(0)

    # Rimuovi colonne con varianza zero
    X = X.loc[:, X.std() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns.tolist()


def prepare_training_test_sets(df_train: pd.DataFrame, df_test: pd.DataFrame, groups: list = None) -> tuple:
    """
    Funzione centralizzata per pulizia e scaling di train e test set.
    Garantisce che entrambi abbiano le stesse colonne e lo stesso scaling.
    """
    # 1. Identifica colonne stilometriche
    feature_cols = get_feature_columns(df_train, groups)

    # 2. Mantieni solo colonne presenti in entrambi
    common_cols = [c for c in feature_cols if c in df_test.columns]

    # 3. Filtra varianza zero sul training
    X_train_raw = df_train[common_cols].fillna(0)
    X_train_raw = X_train_raw.loc[:, X_train_raw.std() > 0]
    valid_cols = X_train_raw.columns.tolist()

    # 4. Prepara dati finali
    X_train_final = X_train_raw[valid_cols]
    X_test_final = df_test[valid_cols].fillna(0)

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    return X_train_scaled, X_test_scaled, scaler, valid_cols


def run_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    """Esegue PCA e restituisce le coordinate e la varianza spiegata."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    variance_ratio = pca.explained_variance_ratio_

    return X_pca, variance_ratio, pca


def run_tsne(X_scaled: np.ndarray, perplexity: int = 5) -> np.ndarray:
    """Esegue t-SNE per visualizzazione."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne


def run_lda(X_scaled: np.ndarray, y: np.ndarray, n_components: int = 3) -> tuple:
    """
    Esegue LDA (Linear Discriminant Analysis) - riduzione dimensionale SUPERVISIONATA.

    A differenza di PCA/UMAP che cercano varianza/struttura generale,
    LDA cerca le direzioni che MASSIMIZZANO la separazione tra classi.

    Parameters:
    -----------
    X_scaled : features scalate
    y : etichette (es. ['llama', 'mistral', ...])
    n_components : dimensioni output (max = n_classi - 1)

    Returns:
    --------
    X_lda : coordinate trasformate
    lda : modello LDA (contiene explained_variance_ratio_)
    """
    n_classes = len(np.unique(y))
    # LDA può avere al massimo n_classes - 1 componenti
    n_components = min(n_components, n_classes - 1)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)

    return X_lda, lda


def run_lda_analysis(df_metrics: pd.DataFrame,
                     topic: str = None,
                     groups: list = None,
                     show_plot: bool = True) -> dict:
    """
    Esegue LDA e visualizza la separazione tra modelli.

    LDA è SUPERVISIONATO: usa le etichette per trovare la proiezione
    che massimizza la separazione tra classi (modelli).

    Parameters:
    -----------
    df_metrics : DataFrame con features stilometriche
    topic : str o None - filtra per topic specifico
    groups : lista gruppi features
    show_plot : se True, mostra il plot

    Returns:
    --------
    dict con X_lda, lda model, variance explained, etc.
    """

    # 1. Filtra per topic
    if topic is not None:
        df_subset = df_metrics[df_metrics['topic'] == topic].copy()
        title_suffix = f" - Topic: {topic}"
    else:
        df_subset = df_metrics.copy()
        title_suffix = " - Tutti i topic"

    n_samples = len(df_subset)
    modelli = df_subset['modello'].unique()
    n_modelli = len(modelli)

    print(f"{'='*60}")
    print(f"LDA (Linear Discriminant Analysis){title_suffix}")
    print(f"{'='*60}")
    print(f"Campioni: {n_samples}")
    print(f"Modelli: {modelli.tolist()}")

    # 2. Prepara features
    X_scaled, feature_names = prepare_features(df_subset, groups=groups)
    y = df_subset['modello'].values
    print(f"Features: {len(feature_names)}")

    # 3. LDA
    n_components = min(2, n_modelli - 1)
    X_lda, lda = run_lda(X_scaled, y, n_components=n_components)

    # Varianza spiegata
    var_ratio = lda.explained_variance_ratio_
    print(f"\nComponenti LDA: {n_components}")
    print(f"Varianza spiegata: {var_ratio}")
    print(f"Totale: {sum(var_ratio):.1%}")

    # 4. Plot
    if show_plot:
        if n_components == 1:
            # Solo 1 componente: plot 1D con jitter
            fig, ax = plt.subplots(figsize=(12, 4))
            for i, modello in enumerate(modelli):
                mask = df_subset['modello'] == modello
                color = MODEL_COLORS.get(modello, 'gray')
                jitter = np.random.normal(0, 0.1, mask.sum())
                ax.scatter(X_lda[mask, 0], jitter, c=color, label=modello,
                          s=150, alpha=0.7, edgecolors='black')
            ax.set_xlabel(f'LD1 ({var_ratio[0]:.1%} varianza)')
            ax.set_ylabel('(jitter per visualizzazione)')
            ax.set_title(f'LDA{title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # 2 componenti: plot 2D
            fig, ax = plt.subplots(figsize=(10, 7))
            for modello in modelli:
                mask = df_subset['modello'] == modello
                color = MODEL_COLORS.get(modello, 'gray')
                ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=modello,
                          s=150, alpha=0.7, edgecolors='black')
            ax.set_xlabel(f'LD1 ({var_ratio[0]:.1%})')
            ax.set_ylabel(f'LD2 ({var_ratio[1]:.1%})')
            ax.set_title(f'LDA - Separazione Supervisionata{title_suffix}\n'
                        f'(Proiezione che massimizza separazione tra modelli)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # 5. Calcola centroidi per ogni modello
    centroids = {}
    for modello in modelli:
        mask = df_subset['modello'] == modello
        centroids[modello] = X_lda[mask].mean(axis=0)

    print(f"\nCentroidi nello spazio LDA:")
    for modello, centroid in centroids.items():
        print(f"  {modello}: {centroid.round(3)}")

    return {
        'topic_filter': topic,
        'X_lda': X_lda,
        'lda': lda,
        'variance_ratio': var_ratio,
        'centroids': centroids,
        'n_samples': n_samples,
        'feature_names': feature_names
    }


def compare_lda_vs_pca(df_metrics: pd.DataFrame, groups: list = None) -> None:
    """
    Confronta visivamente PCA (non supervisionato) vs LDA (supervisionato).
    Mostra come LDA separa meglio i modelli.
    """

    X_scaled, _ = prepare_features(df_metrics, groups=groups)
    y = df_metrics['modello'].values
    modelli = df_metrics['modello'].unique()

    # PCA
    X_pca, var_pca, _ = run_pca(X_scaled, n_components=2)

    # LDA
    X_lda, lda = run_lda(X_scaled, y, n_components=2)
    var_lda = lda.explained_variance_ratio_

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    for modello in modelli:
        mask = df_metrics['modello'] == modello
        color = MODEL_COLORS.get(modello, 'gray')
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=modello,
                       s=120, alpha=0.7, edgecolors='black')
    axes[0].set_xlabel(f'PC1 ({var_pca[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({var_pca[1]:.1%})')
    axes[0].set_title('PCA (Non Supervisionato)\nCerca varianza, trova TOPIC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LDA
    for modello in modelli:
        mask = df_metrics['modello'] == modello
        color = MODEL_COLORS.get(modello, 'gray')
        axes[1].scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=modello,
                       s=120, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel(f'LD1 ({var_lda[0]:.1%})')
    axes[1].set_ylabel(f'LD2 ({var_lda[1]:.1%})')
    axes[1].set_title('LDA (Supervisionato)\nCerca separazione, trova MODELLI')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('PCA vs LDA: Non Supervisionato vs Supervisionato', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_umap(X_scaled: np.ndarray, metric: str = 'cosine', n_neighbors: int = 15, n_components: int = 2) -> np.ndarray:
    """Esegue UMAP per visualizzazione."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42, metric=metric)
    return reducer.fit_transform(X_scaled)


def run_supervised_umap(X_scaled: np.ndarray, y: np.ndarray,
                        n_neighbors: int = 15, n_components: int = 2,
                        metric: str = 'cosine') -> tuple:
    """
    Esegue UMAP SUPERVISIONATO - usa le etichette per guidare l'embedding.

    A differenza di UMAP standard (non supervisionato), UMAP supervisionato
    usa le etichette y per creare un embedding che separa meglio le classi.

    Parameters:
    -----------
    X_scaled : features scalate
    y : etichette (es. ['llama', 'mistral', ...])
    n_neighbors : numero di vicini (default 15)
    n_components : dimensioni output (default 2)
    metric : metrica distanza (default 'cosine')

    Returns:
    --------
    X_umap : coordinate trasformate
    reducer : modello UMAP (per transform su nuovi dati)
    """

    # Codifica etichette in numeri
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # UMAP supervisionato
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=42,
        metric=metric,
        target_metric='categorical'  # per etichette categoriche
    )
    X_umap = reducer.fit_transform(X_scaled, y=y_encoded)

    return X_umap, reducer, le


def run_knn(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray = None, y_test: np.ndarray = None,
            n_neighbors: int = 5, cv: int = 5) -> dict:
    """
    Esegue classificazione con K-Nearest Neighbors.

    KNN è utile su embedding UMAP: classifica basandosi sulla vicinanza
    nello spazio ridotto (i vicini più simili "votano" per la classe).

    Parameters:
    -----------
    X_train : features di training
    y_train : etichette di training
    X_test : features di test (opzionale)
    y_test : etichette di test (opzionale)
    n_neighbors : numero di vicini per KNN (default 5)
    cv : fold per cross-validation

    Returns:
    --------
    dict con accuracy, predictions, confusion_matrix, etc.
    """

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Cross-validation su training
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv)

    # Fit su tutto il training
    knn.fit(X_train, y_train)

    result = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'classifier': knn,
        'classes': knn.classes_
    }

    # Se fornito test set, valuta
    if X_test is not None and y_test is not None:

        y_pred = knn.predict(X_test)
        result['test_accuracy'] = accuracy_score(y_test, y_pred)
        result['test_predictions'] = y_pred
        result['test_confusion_matrix'] = confusion_matrix(y_test, y_pred, labels=knn.classes_)
        result['test_classification_report'] = classification_report(y_test, y_pred, output_dict=True)

    return result


def run_supervised_umap_analysis(df_metrics: pd.DataFrame,
                                  topic: str = None,
                                  groups: list = None,
                                  n_neighbors: int = 15,
                                  knn_neighbors: int = 5,
                                  cv: int = 5,
                                  show_plot: bool = True) -> dict:
    """
    Esegue UMAP supervisionato e valuta con KNN.

    Parameters:
    -----------
    df_metrics : DataFrame con features stilometriche
    topic : str o None - filtra per topic specifico
    groups : lista gruppi features
    n_neighbors : parametro UMAP (vicini per embedding)
    knn_neighbors : parametro KNN (vicini per classificazione)
    cv : fold per cross-validation KNN
    show_plot : se True, mostra il plot

    Returns:
    --------
    dict con X_umap, reducer, knn_accuracy, ari, etc.
    """

    # 1. Filtra per topic
    if topic is not None:
        df_subset = df_metrics[df_metrics['topic'] == topic].copy()
        title_suffix = f" - Topic: {topic}"
    else:
        df_subset = df_metrics.copy()
        title_suffix = " - Tutti i topic"

    n_samples = len(df_subset)
    modelli = df_subset['modello'].unique()

    print(f"{'='*60}")
    print(f"UMAP SUPERVISIONATO{title_suffix}")
    print(f"{'='*60}")
    print(f"Campioni: {n_samples}")
    print(f"Modelli: {modelli.tolist()}")

    # 2. Prepara features
    X_scaled, feature_names = prepare_features(df_subset, groups=groups)
    y = df_subset['modello'].values
    print(f"Features: {len(feature_names)}")

    # 3. UMAP supervisionato
    X_umap, reducer, le = run_supervised_umap(X_scaled, y, n_neighbors=n_neighbors)

    # 4. KNN per classificazione sull'embedding UMAP
    knn_results = run_knn(X_umap, y, n_neighbors=knn_neighbors, cv=cv)
    knn_accuracy = knn_results['cv_accuracy_mean']
    knn_std = knn_results['cv_accuracy_std']

    # 5. Clustering per calcolare ARI (per confronto)
    clusters, _ = run_kmeans(X_umap, n_clusters=len(modelli))
    ari = adjusted_rand_score(y, clusters)

    print(f"\nUMAP n_neighbors: {n_neighbors}")
    print(f"KNN (k={knn_neighbors}, cv={cv}): {knn_accuracy:.2%} (+/- {knn_std:.2%})")
    print(f"ARI (clustering): {ari:.3f}")

    # 6. Plot
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 7))

        for modello in modelli:
            mask = df_subset['modello'] == modello
            color = MODEL_COLORS.get(modello, 'gray')
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=color, label=modello, s=150, alpha=0.7, edgecolors='black')

        ax.set_xlabel('UMAP1 (supervisionato)')
        ax.set_ylabel('UMAP2 (supervisionato)')
        ax.set_title(f'UMAP Supervisionato{title_suffix}\nKNN Accuracy={knn_accuracy:.1%} | ARI={ari:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        'topic_filter': topic,
        'X_umap': X_umap,
        'reducer': reducer,
        'label_encoder': le,
        'knn_accuracy': knn_accuracy,
        'knn_std': knn_std,
        'knn_results': knn_results,
        'ari': ari,
        'clusters': clusters,
        'n_samples': n_samples,
        'feature_names': feature_names,
        'df_subset': df_subset
    }


def run_kmeans(X: np.ndarray, n_clusters: int = 4) -> tuple:
    """Esegue K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def analyze_clusters(X_reduced: np.ndarray, df_metrics: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Analizza se i cluster si formano per modello o per topic.
    Restituisce ARI e crosstab per entrambi.
    """
    cluster_labels, _ = run_kmeans(X_reduced, n_clusters=n_clusters)

    results = {
        'clusters': cluster_labels,
        'ari_modello': adjusted_rand_score(df_metrics['modello'], cluster_labels),
        'crosstab_modello': pd.crosstab(df_metrics['modello'], cluster_labels, margins=True)
    }

    if 'topic' in df_metrics.columns:
        results['ari_topic'] = adjusted_rand_score(df_metrics['topic'], cluster_labels)
        results['crosstab_topic'] = pd.crosstab(df_metrics['topic'], cluster_labels, margins=True)

    return results


def run_classification(X: np.ndarray, y: np.ndarray, cv: int = 5, return_details: bool = False) -> dict:
    """
    Esegue classificazione con Random Forest e cross-validation.

    Come funziona:
    1. Crea un Random Forest con 100 alberi decisionali
    2. Esegue cross-validation stratificata (mantiene proporzioni classi)
    3. Calcola accuracy media e deviazione standard
    4. Fa un fit finale su tutti i dati per estrarre feature importance

    Parameters:
    -----------
    X : array di features (n_samples, n_features)
    y : array di etichette (n_samples,) - es. ['llama', 'mistral', ...]
    cv : numero di fold per cross-validation
    return_details : se True, restituisce anche predictions e classification_report

    Returns:
    --------
    dict con accuracy_mean, accuracy_std, scores, classifier, [details]
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation: divide i dati in cv parti, allena su cv-1 e testa su 1
    scores = cross_val_score(clf, X, y, cv=cv)

    # Fit finale su tutti i dati (per feature importance)
    clf.fit(X, y)

    result = {
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std(),
        'scores': scores,
        'classifier': clf,
        'classes': clf.classes_  # Lista dei modelli che sa riconoscere
    }

    if return_details:
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        result['predictions'] = y_pred
        result['confusion_matrix'] = confusion_matrix(y, y_pred, labels=clf.classes_)
        result['classification_report'] = classification_report(y, y_pred, output_dict=True)

    return result


def run_classification_by_topic(df_metrics: pd.DataFrame, groups: list = None, cv: int = 3) -> pd.DataFrame:
    """
    Esegue classificazione separatamente per ogni topic.
    Permette di capire se i modelli sono distinguibili all'interno di ogni genere.

    Parameters:
    -----------
    df_metrics : DataFrame con features stilometriche + colonne 'modello' e 'topic'
    groups : lista di gruppi di features da usare (es. ['POS', 'Lessico'])
    cv : numero di fold per cross-validation

    Returns:
    --------
    DataFrame con accuracy per ogni topic
    """
    results = []

    topics = df_metrics['topic'].unique() if 'topic' in df_metrics.columns else ['all']

    for topic in topics:
        if topic == 'all':
            df_subset = df_metrics
        else:
            df_subset = df_metrics[df_metrics['topic'] == topic]

        if len(df_subset) < cv * 2:  # Troppo pochi campioni
            results.append({
                'topic': topic,
                'n_samples': len(df_subset),
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'note': 'Campioni insufficienti'
            })
            continue

        X, feature_names = prepare_features(df_subset, groups=groups)
        y = df_subset['modello'].values

        try:
            res = run_classification(X, y, cv=cv)
            results.append({
                'topic': topic,
                'n_samples': len(df_subset),
                'n_features': X.shape[1],
                'accuracy_mean': res['accuracy_mean'],
                'accuracy_std': res['accuracy_std'],
                'scores': res['scores'].tolist()
            })
        except Exception as e:
            results.append({
                'topic': topic,
                'n_samples': len(df_subset),
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'note': str(e)
            })

    return pd.DataFrame(results)


def full_analysis(df_metrics: pd.DataFrame, groups: list = None, cv: int = 3) -> dict:
    """
    Esegue un'analisi completa: clustering + classificazione su tutti i dati e per topic.

    Returns:
    --------
    dict con tutti i risultati dell'analisi
    """
    X_scaled, feature_names = prepare_features(df_metrics, groups=groups)
    y = df_metrics['modello'].values
    n_modelli = len(df_metrics['modello'].unique())

    results = {
        'n_samples': len(df_metrics),
        'n_features': len(feature_names),
        'n_modelli': n_modelli,
        'feature_names': feature_names
    }

    # 1. Clustering analysis
    results['clustering'] = {}
    for method_name, X_reduced in [
        ('PCA', run_pca(X_scaled, n_components=2)[0]),
        ('UMAP', run_umap(X_scaled)),
    ]:
        analysis = analyze_clusters(X_reduced, df_metrics, n_clusters=n_modelli)
        results['clustering'][method_name] = {
            'ari_modello': analysis['ari_modello'],
            'ari_topic': analysis.get('ari_topic', None)
        }

    # 2. Classification globale
    clf_results = run_classification(X_scaled, y, cv=cv, return_details=True)
    results['classification_global'] = {
        'accuracy_mean': clf_results['accuracy_mean'],
        'accuracy_std': clf_results['accuracy_std'],
        'confusion_matrix': clf_results['confusion_matrix'],
        'classes': clf_results['classes'].tolist()
    }

    # 3. Classification per topic
    results['classification_by_topic'] = run_classification_by_topic(df_metrics, groups=groups, cv=cv)

    # 4. Feature importance
    importance = get_feature_importance(clf_results['classifier'], feature_names, top_n=20)
    results['top_features'] = importance

    return results


def get_feature_importance(clf, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Restituisce le features più importanti per la classificazione."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance.head(top_n)


def compare_models_by_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Confronta le medie delle features tra modelli."""
    results = []

    for feat in features:
        if feat in df.columns:
            for model in df['modello'].unique():
                mean_val = df.loc[df['modello'] == model, feat].mean()
                results.append({
                    'feature': feat,
                    'modello': model,
                    'mean': mean_val
                })

    df_comparison = pd.DataFrame(results)
    return df_comparison.pivot(index='feature', columns='modello', values='mean')


def run_clustering_analysis(df_metrics: pd.DataFrame,
                            topic: str = None,
                            groups: list = None,
                            method: str = 'umap',
                            n_neighbors: int = 15,
                            show_plot: bool = True) -> dict:
    """
    Esegue clustering su tutti i topic o su un singolo topic.

    Parameters:
    -----------
    df_metrics : DataFrame con features stilometriche
    topic : str o None
        - None → usa tutti i topic
        - 'narrative' | 'argumentation' | 'dialogue' | 'description' → filtra
    groups : lista gruppi features (es. ['POS', 'Lessico'])
    method : 'umap' | 'pca' | 'tsne'
    n_neighbors : parametro per UMAP
    show_plot : se True, mostra il plot

    Returns:
    --------
    dict con:
        - 'df_subset': dati filtrati
        - 'X_reduced': coordinate 2D
        - 'clusters': etichette cluster
        - 'ari_modello': ARI rispetto ai modelli
        - 'crosstab_modello': crosstab modello vs cluster

    Esempi:
    -------
    # Tutti i topic insieme
    results = run_clustering_analysis(df_metrics)

    # Solo narrative
    results = run_clustering_analysis(df_metrics, topic='narrative')

    # Solo dialogue con PCA
    results = run_clustering_analysis(df_metrics, topic='dialogue', method='pca')
    """

    # 1. Filtra per topic
    if topic is not None:
        if topic not in df_metrics['topic'].unique():
            raise ValueError(f"Topic '{topic}' non trovato. Disponibili: {df_metrics['topic'].unique().tolist()}")
        df_subset = df_metrics[df_metrics['topic'] == topic].copy()
        title_suffix = f" - Topic: {topic}"
    else:
        df_subset = df_metrics.copy()
        title_suffix = " - Tutti i topic"

    n_samples = len(df_subset)
    n_modelli = len(df_subset['modello'].unique())

    print(f"{'='*50}")
    print(f"CLUSTERING ANALYSIS{title_suffix}")
    print(f"{'='*50}")
    print(f"Campioni: {n_samples}")
    print(f"Modelli: {df_subset['modello'].unique().tolist()}")
    if topic is None:
        print(f"Topic: {df_subset['topic'].unique().tolist()}")

    # 2. Prepara features
    X_scaled, feature_names = prepare_features(df_subset, groups=groups)
    print(f"Features: {len(feature_names)}")

    # 3. Riduzione dimensionale
    if method == 'umap':
        X_reduced = run_umap(X_scaled, n_neighbors=n_neighbors)
        method_label = f'UMAP (n_neighbors={n_neighbors})'
    elif method == 'pca':
        X_reduced, var_ratio, _ = run_pca(X_scaled, n_components=2)
        method_label = f'PCA (var={sum(var_ratio):.1%})'
    elif method == 'tsne':
        perp = min(5, n_samples // 4)  # perplexity adattiva
        X_reduced = run_tsne(X_scaled, perplexity=perp)
        method_label = f't-SNE (perp={perp})'
    else:
        raise ValueError(f"Method deve essere 'umap', 'pca' o 'tsne'")

    # 4. Clustering
    clusters, kmeans = run_kmeans(X_reduced, n_clusters=n_modelli)

    # 5. Calcola ARI
    ari_modello = adjusted_rand_score(df_subset['modello'], clusters)
    crosstab = pd.crosstab(df_subset['modello'], clusters, margins=True)

    print(f"\n{method_label}")
    print(f"ARI per MODELLO: {ari_modello:.3f}")
    print(f"\nCrosstab Modello vs Cluster:")
    print(crosstab)

    # 6. Plot
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 7))

        for modello in df_subset['modello'].unique():
            mask = df_subset['modello'] == modello
            color = MODEL_COLORS.get(modello, 'gray')
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                      c=color, label=modello, s=150, alpha=0.7, edgecolors='black')

        ax.set_xlabel(f'{method.upper()}1')
        ax.set_ylabel(f'{method.upper()}2')
        ax.set_title(f'{method_label}{title_suffix}\nARI={ari_modello:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        'topic_filter': topic,
        'df_subset': df_subset,
        'X_reduced': X_reduced,
        'clusters': clusters,
        'n_samples': n_samples,
        'n_features': len(feature_names),
        'method': method,
        'ari_modello': ari_modello,
        'crosstab_modello': crosstab
    }


def run_clustering_all_topics(df_metrics: pd.DataFrame,
                               groups: list = None,
                               method: str = 'umap') -> pd.DataFrame:
    """
    Esegue clustering separatamente per ogni topic e confronta i risultati.

    Returns:
    --------
    DataFrame con ARI per ogni topic
    """
    results = []

    # Prima tutti insieme
    res_all = run_clustering_analysis(df_metrics, topic=None, groups=groups,
                                       method=method, show_plot=False)
    results.append({
        'topic': 'ALL',
        'n_samples': res_all['n_samples'],
        'ari_modello': res_all['ari_modello']
    })

    # Poi ogni topic
    for topic in df_metrics['topic'].unique():
        res = run_clustering_analysis(df_metrics, topic=topic, groups=groups,
                                       method=method, show_plot=False)
        results.append({
            'topic': topic,
            'n_samples': res['n_samples'],
            'ari_modello': res['ari_modello']
        })

    df_results = pd.DataFrame(results)

    print("\n" + "="*50)
    print("RIEPILOGO CLUSTERING PER TOPIC")
    print("="*50)
    print(df_results.to_string(index=False))

    return df_results


def salva_classificatore(clf, scaler, valid_cols, filename_prefix="random_forest"):
    """
    Salva il classificatore, lo scaler e le colonne valide su disco
    per usarle in futuro su nuovi testi.
    """
    import os
    import joblib
    
    os.makedirs('models', exist_ok=True)
    
    clf_path = f'models/{filename_prefix}_model.pkl'
    scaler_path = f'models/{filename_prefix}_scaler.pkl'
    cols_path = f'models/{filename_prefix}_cols.pkl'
    
    joblib.dump(clf, clf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(valid_cols, cols_path)
    
    print("="*50)
    print("MODELLO SALVATO CON SUCCESSO")
    print("="*50)
    print(f"Modello: {clf_path}")
    print(f"Scaler:  {scaler_path}")
    print(f"Colonne: {cols_path}")
    
    return clf_path, scaler_path, cols_path
