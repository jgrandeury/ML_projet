# -*- coding: utf-8 -*-
"""app.py

Application d'analyse des utilisateurs avec Streamlit pour Google Colab
"""

# Installer les packages nécessaires

# Importation des bibliothèques
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Enregistrer le code de l'application dans un fichier app.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Titre de l'application
st.title("Tableau de Bord d'Analyse des Utilisateurs")

# --- Chargement et Préparation des Données ---
try:
    df = pd.read_csv("owa_action_fact.csv")
except FileNotFoundError:
    st.error("Le fichier owa_action_fact.csv n'a pas été trouvé. Veuillez vous assurer qu'il se trouve dans le même répertoire que ce script dans Colab.")
    st.stop()

cols_to_drop = [col for col in df.columns if col.endswith('_id') and col not in ['id', 'visitor_id']]
df_cleaned = df.drop(columns=cols_to_drop)

engagement_map = {
    'publish': 2,
    'animate': 2,
    'participate': 1,
    'reaction': 1,
    'user': 0
}
df_cleaned['engagement_score'] = df_cleaned['action_group'].map(engagement_map).fillna(0)

engagement_df = df_cleaned.groupby("visitor_id").agg(
    avg_engagement_score=('engagement_score', 'mean')
).reset_index()

engagement_df['engagement_class'] = engagement_df['avg_engagement_score']

df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], errors='coerce')
df_cleaned['hour'] = df_cleaned['timestamp'].dt.hour

agg_df = df_cleaned.groupby("visitor_id").agg(
    total_actions=('id', 'count'),
    unique_action_names=('action_name', 'nunique'),
    unique_action_groups=('action_group', 'nunique'),
    avg_days_since_prior_session=('days_since_prior_session', 'mean'),
    avg_days_since_first_session=('days_since_first_session', 'mean'),
    total_sessions=('num_prior_sessions', 'sum'),
    median_days_since_prior_session=('days_since_prior_session', 'median'),
    max_days_since_first_session=('days_since_first_session', 'max'),
    std_days_since_prior_session=('days_since_prior_session', 'std'),
    is_repeat_visitor=('is_repeat_visitor', 'max'),
    is_new_visitor=('is_new_visitor', 'max'),
    lang_count=('language', 'nunique'),
    unique_mediums=('medium', 'nunique'),
    avg_hour_of_activity=('hour', 'mean')
).reset_index()

agg_df['high_activity'] = (agg_df['total_actions'] > 100).astype(int)
agg_df['very_high_activity'] = (agg_df['total_actions'] > 300).astype(int)
agg_df['many_sessions'] = (agg_df['total_sessions'] > 10).astype(int)
agg_df['super_sessions'] = (agg_df['total_sessions'] > 30).astype(int)
agg_df['diverse_actions'] = (agg_df['unique_action_names'] > 5).astype(int)
agg_df['ultra_diverse_actions'] = (agg_df['unique_action_names'] > 10).astype(int)

agg_df = pd.merge(agg_df, engagement_df[['visitor_id', 'engagement_class']], on='visitor_id')

action_counts = df_cleaned.pivot_table(
    index='visitor_id',
    columns='action_group',
    values='id',
    aggfunc='count',
    fill_value=0
).reset_index()

features_df = pd.merge(agg_df, action_counts, on='visitor_id')

X_clust = features_df.drop(['visitor_id', 'engagement_class'], axis=1)
X_clust = X_clust.fillna(X_clust.median(numeric_only=True))
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_clust_scaled)
features_df['cluster'] = kmeans_clusters

# --- Sidebar pour la navigation ---
st.sidebar.header("Navigation")
selected_tab = st.sidebar.radio("Aller à", ["Aperçu Général", "Analyse des Clusters", "Prédiction d'Engagement", "Recommandations"])

# Recommandations spécifiques par cluster
recommandations = {
    0: "Relance ciblée via email avec mise en avant de fonctionnalités \"en un clic\". "
       "Suppression de frictions : tutoriel simple ou onboarding ultra light. "
       "Campagnes d’activation avec récompense à la première action.",
    1: "Relancer avec du contenu personnalisé ou recommandations basées sur leurs usages. "
       "Emails “Vous aimerez aussi…” basés sur les premières actions effectuées. "
       "Créer un parcours d’engagement progressif (avec jalons visibles).",
    2: "Mettre en avant des contenus plus experts ou des outils avancés. "
       "Proposer une implication communautaire : tutoriels, retours, sondages. "
       "Cibler avec des relances pour les faire passer au niveau “super”.",
    3: "Valorisation de leur fidélité (badges, classements). "
       "Invitations à participer à des tests utilisateurs ou sondages. "
       "Contenus plus approfondis adaptés à leur usage avancé.",
    4: "Intégration au programme ambassadeur ou mentorat. "
       "Accès prioritaire aux nouveautés. "
       "Sollicitations pour co-créer des contenus ou animer la communauté."
}

# --- Onglet Aperçu Général ---
if selected_tab == "Aperçu Général":
    st.subheader("Statistiques Générales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre Total d'Utilisateurs", features_df['visitor_id'].nunique())
    col2.metric("Nombre Moyen d'Actions par Utilisateur", f"{features_df['total_actions'].mean():.2f}")
    col3.metric("Nombre Moyen de Sessions par Utilisateur", f"{features_df['total_sessions'].mean():.2f}")

    st.subheader("Distribution des Classes d'Engagement")
    engagement_counts = engagement_df['engagement_class'].value_counts().sort_index()
    st.bar_chart(engagement_counts)

    st.subheader("Distribution des Clusters d'Utilisateurs")
    cluster_counts = features_df['cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

# --- Onglet Analyse des Clusters ---
elif selected_tab == "Analyse des Clusters":
    st.subheader("Visualisation des Clusters (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_clust_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=['Composante 1', 'Composante 2'])
    pca_df['Cluster'] = features_df['cluster']

    fig_pca = px.scatter(pca_df, x='Composante 1', y='Composante 2', color='Cluster',
                         title="Clusters d'utilisateurs visualisés par PCA",
                         color_continuous_scale=px.colors.qualitative.Set2)
    st.plotly_chart(fig_pca)

    st.subheader("Profil des Clusters")
    cluster_profiles = features_df.groupby('cluster').mean().T
    st.write(cluster_profiles)

    st.subheader("Importance des Caractéristiques pour la Formation des Clusters")
    cluster_means = features_df.groupby('cluster').mean()
    overall_mean = features_df.drop('cluster', axis=1).mean()
    feature_importance = ((cluster_means - overall_mean)**2).mean(axis=0).sort_values(ascending=False)
    st.bar_chart(feature_importance)

# --- Onglet Prédiction d'Engagement ---
elif selected_tab == "Prédiction d'Engagement":
    st.subheader("Prédiction de la Classe d'Engagement")
    st.write("Ici, nous allons construire un modèle pour prédire la classe d'engagement d'un utilisateur.")

    X_pred = features_df.drop(['visitor_id', 'engagement_class', 'cluster'], axis=1)
    y_pred = features_df['engagement_class'].round().astype(int)

    X_pred = X_pred.fillna(X_pred.median(numeric_only=True))
    scaler_pred = StandardScaler()
    X_pred_scaled = scaler_pred.fit_transform(X_pred)

    X_train, X_test, y_train, y_test = train_test_split(X_pred_scaled, y_pred, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    y_pred_classes = np.round(y_pred_model).astype(int)

    st.subheader("Performance du Modèle de Prédiction")
    st.text("Rapport de Classification:")
    st.text(classification_report(y_test, y_pred_classes, zero_division=0))

    st.text("Matrice de Confusion:")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.colorbar(disp)
    classes = np.unique(y_pred)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Valeurs')
    thresh = cm.max() / 2.
    for i, j in np.ndenumerate(cm):
        plt.text(j, i, str(j), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    st.pyplot(fig_cm)

    st.subheader("Importance des Caractéristiques pour la Prédiction")
    feature_importances = pd.Series(model.feature_importances_, index=X_pred.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    st.bar_chart(feature_importances_sorted)

    st.subheader("Prédire l'Engagement d'un Nouvel Utilisateur")
    with st.form("new_user_prediction"):
        new_user_data = {}
        for col in X_pred.columns:
            if X_pred[col].dtype in ['int64', 'float64']:
                new_user_data[col] = st.number_input(f"{col} (moyenne: {X_pred[col].mean():.2f})")
            elif X_pred[col].dtype == 'object':
                unique_vals = X_pred[col].unique().tolist()
                new_user_data[col] = st.selectbox(f"{col}", unique_vals)
            else:
                new_user_data[col] = st.text_input(f"{col}")

        predict_button = st.form_submit_button("Prédire l'Engagement")

        if predict_button:
            new_user_df = pd.DataFrame([new_user_data])
            for col in X_pred.columns:
                if col not in new_user_df.columns:
                    new_user_df[col] = X_pred[col].median()

            new_user_scaled = scaler_pred.transform(new_user_df[X_pred.columns])
            predicted_engagement = model.predict(new_user_scaled)[0]
            predicted_class = round(predicted_engagement)
            st.write(f"La classe d'engagement prédite pour ce nouvel utilisateur est : **{predicted_class}**")
# --- Onglet Recommandations Basées sur les Clusters ---
elif selected_tab == "Recommandations":
    st.sidebar.header("Recommandations")
    st.sidebar.subheader("Basées sur les Clusters")

    if "cluster" in features_df.columns:
        selected_cluster = st.sidebar.selectbox("Sélectionner un Cluster", sorted(features_df['cluster'].unique()))
        st.subheader(f"Recommandations pour le Cluster {selected_cluster}")

        cluster_data = features_df[features_df['cluster'] == selected_cluster]
        if not cluster_data.empty:
            avg_cluster_engagement = cluster_data['engagement_class'].mean()
            st.write(f"Niveau d'engagement moyen du cluster : **{avg_cluster_engagement:.2f}**")

            action_cols = [col for col in features_df.columns if col in engagement_map.keys()]
            cluster_action_means = cluster_data[action_cols].mean()
            overall_action_means = features_df[action_cols].mean()
            positive_deviations = (cluster_action_means - overall_action_means).sort_values(ascending=False)
            negative_deviations = (cluster_action_means - overall_action_means).sort_values(ascending=True)

            st.write("Actions surreprésentées dans ce cluster :")
            st.write(positive_deviations[positive_deviations > 0])

            st.write("Actions sous-représentées dans ce cluster :")
            st.write(negative_deviations[negative_deviations < 0])
            
            st.subheader("Recommandation Stratégique pour ce Cluster")
            if selected_cluster in recommandations:
                st.success(recommandations[selected_cluster])
            else:
                st.info("Aucune recommandation spécifique définie pour ce cluster.")


#            st.info("Recommandations potentielles pour ce cluster :")
#            if positive_deviations.any():
#                top_positive_actions = positive_deviations[positive_deviations > 0].index.tolist()
#               st.write(f"- Mettre en avant davantage les contenus/fonctionnalités liés à : **{', '.join(top_positive_actions)}** pour maintenir l'engagement.")
#            if negative_deviations.any():
#                top_negative_actions = negative_deviations[negative_deviations < 0].index.tolist()
#                st.write(f"- Explorer des stratégies pour encourager davantage d'interactions avec : **{', '.join(top_negative_actions)}**.")

        else:
            st.warning("Aucun utilisateur trouvé dans ce cluster.")
    else:
        st.sidebar.warning("Le clustering n'a pas été effectué. Veuillez exécuter l'analyse des clusters.")

# Après avoir sauvegardé le fichier, configurez ngrok et lancez l'application
# Configurer le port pour Streamlit
port = 8501
