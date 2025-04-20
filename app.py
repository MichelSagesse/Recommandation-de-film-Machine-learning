import streamlit as st
import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import requests
import json
from streamlit_lottie import st_lottie
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from PIL import Image
#Configuration de la page de la barre de recheche
st.set_page_config(
    page_title="Home",
    layout="wide",
    page_icon="🏠")
#Mise en page du fond
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/photos-premium/motif-simple-fond-papier-blanc-ordinaire-vintage_136558-43305.jpg");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
#horizontal menu with custom style
def streamlit_menu(): 
        selected = option_menu(
            menu_title=None,  # required
            options=["Accueil", "Modèles", "Contact"],  # required
            icons=["house", "buildings-fill", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#ff6e27"},
            },
        )
        return selected


selected = streamlit_menu()

if selected == "Accueil":
    st.header(f"Bienvenue sur la page d'accueil de nos deux applications de recommandation de films.")
if selected == "Modèles":
    st.header(f"Nous avons deux modèles")
    st.write("1)K-nearest neighbors(KNN): L’algorithme des k plus proches voisins (kNN) est un algorithme de Machine Learning supervisé simple qui peut être utilisé pour résoudre des problèmes de classification et de régression. Il est facile à mettre en œuvre et à comprendre, mais présente un inconvénient majeur: son ralentissement est important à mesure que la taille des données utilisées augmente.kNN recherche les distances entre une requête cible et toutes les observations des données. Ensuite il sélectionne les k observations les plus proches de la requête. Enfin il vote pour le libellé le plus fréquent (dans le cas de la classification) ou la moyenne des libellés (en le cas de la régression).")
    st.write("2)Modèle de recommandation de contenu basé sur la similarité de texte: Un modèle de recommandation de contenu basé sur la similarité de texte est un système qui recommande des éléments similaires en se basant sur le contenu textuel des éléments eux-mêmes. Ce type de modèle est largement utilisé dans divers domaines tels que le commerce électronique, la musique, les films, les articles de presse, etc.")
if selected == "Contact":
    # Use local CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css(r"C:\Users\user\Downloads\Projet ML\pages\style\style.css")
    #
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/michelsagesse16@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

st.markdown(page_bg_img, unsafe_allow_html=True)
# IMPORTER LES DATAFRAMES UTILISES
#Dataset1
liste_films = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\liste_films.pkl.gz")
liste_genres = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\liste_genres.pkl.gz")
liste_acteurs = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\liste_acteurs.pkl.gz")
liste_annees = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\liste_annees.pkl.gz")
data_machine_learning_clean = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\data_machine_learning_clean.pkl.gz")
#Dataset2
liste_films = pd.read_pickle(r"C:\Users\user\Downloads\Projet ML\Model1(KNN)\Données nettoyées\liste_films.pkl.gz")
new_data = pd.read_csv(r"C:\Users\user\Downloads\Projet ML\Model2 textvectorclassificator(countvectorizer)\Dataset1\new_data.csv")

page = st.sidebar.radio('Go to', ['App_modèle1', 'App_modèle2'])
if page == 'App_modèle1':
    # SUPPRIMER LES DOUBLONS
    data_machine_learning_clean = data_machine_learning_clean.drop_duplicates(subset = "tconst")
    # Réindexer le DataFrame après la suppression des doublons
    data_machine_learning_clean = data_machine_learning_clean.reset_index(drop=True)
    # CONFIGURER LA PAGE
    #Mise en page du fond de la page
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Définition des fonctions de chargement de fichier Lottie
    #def load_lottiefile(filepath: str):
        #with open(filepath, "r") as f:
            #return json.load(f)

    #def load_lottieurl(url: str):
        #r = requests.get(url)
        #if r.status_code != 200:
            #return None
        #return r.json()
    # Appel des deux fonctions
    #lottie_coding = load_lottiefile(r"C:\Users\user\Downloads\Projet ML\pages\image&animation\Animation1.json")  # replace link to local lottie file
    #lottie_hello = load_lottieurl("https://lottie.host/5c39a3f0-4225-4a25-ae7c-a9d6cd9ccbbd/m076uPOTI8.json")
    # Création d'une disposition personnalisée pour aligner l'animation et le titre sur la même ligne
    #col1, col2 = st.columns([1, 2])  # Divise l'espace en 3 colonnes, 1 pour l'animation et 2 pour le titre

    # Affichage de l'animation Lottie dans la première colonne
    #with col1:
        #st_lottie(
            #lottie_hello,
            #speed=1,
            #reverse=False,
            #loop=True,
            #quality="low",  # medium ; high
            #height=150,  # Hauteur de l'animation
            #width=150,  # Largeur de l'animation
        #)

    # Affichage du titre dans la deuxième colonne
    #with col2:
        #st.header('Bienvenue sur notre première page de recommandation de films')

    # LISTES 
    list_film_deroulante_films = ["Saissisez le film que vous aimez"] + list(liste_films["primaryTitle"])
    list_film_deroulante_acteurs = ["Saissisez l'acteur que vous aimez"] + list(liste_acteurs["primaryName"])
    list_film_deroulante_genres = ["Saissisez votre/vos genres"] + list(liste_genres["genres"])

    # REQUETE API 

    site : "https://www.omdbapi.com/"
    #demande de key API : key = "26444a04"
    url_api = "http://www.omdbapi.com/?i="
    key_api = "&apikey=26444a04"
    url_imdb = 'https://www.imdb.com/title/' 

    # MACHINE LEARNING : 

    # Création de la variable X qui prends en variables explicatives toutes les colonnes avec des valeurs boléennes - PAS BESOIN DE SCALER => toutes les lignes et toutes les colonnes à partir d'index 8 
    X = data_machine_learning_clean.iloc[:,8:] 

    # On entraine le modele uniquement sur les 5 voisins les plus proches sur l'ensemble des colonnes choisies (metric = cosine similarity)
    model1 = NearestNeighbors(n_neighbors = 5, metric = "cosine", algorithm = "brute").fit(X)

    # CHOIX DU FILM PAR L'UTILISATEUR

    with st.form("form 1"):
        
        # Indication utilisateur
        st.subheader("1. Recherche par titre de film")
        # Mise en place du choix utilisateurs
        films = st.selectbox("Films : ",list_film_deroulante_films)

        # Bouton submit
        submit_1 = st.form_submit_button("Rechercher")

            
        # PROPOSITION DE L'ALGORITHME

        if submit_1 : 

            # Obtenir tous les renseignements du film
            data_film_choisi = data_machine_learning_clean[(data_machine_learning_clean["primaryTitle"] == films) | (data_machine_learning_clean["originalTitle"] == films) | (data_machine_learning_clean["French_Title"] == films)]

            # On  selectionne seulement les colonnes contenant des booleens sur la ligne du film choisi
            film_choisi = data_film_choisi.iloc[:, 8:]

            #création de la matrice pour rechercher les 4 index des plus proches voisins (dont le film en question)
            distance, indice = model1.kneighbors(film_choisi)

            # Création d'une variable tconst pour récupérer le numéro tconst du film
            tconst = data_machine_learning_clean.iloc[indice[0,1:]]['tconst'].values

            # Création de la liste des suggestions 
            suggestion = data_machine_learning_clean.iloc[indice[0,1:]]['primaryTitle'].values

            # Création de colonnes
            col1 = st.columns(4)

            # Boucle simultanée sur les deux paramètres : suggestion et tconst 
            for films, code_film, colonnes in zip(suggestion,tconst, col1):

                with colonnes:

                    url = f"{url_api}{code_film}{key_api}"  # on remplace tconst par code_film
                    url_imdb2 = f"{url_imdb}{code_film}" 

                    try:
                        response = requests.get(url)
                        #st.write(str(url))       
                        response.raise_for_status()
                        data = response.json()
                        url_image = data['Poster']
                        st.image(url_image, width=200)

                    except requests.exceptions.RequestException as e:
                        print('Une erreur  lors de l\'appel à l\'API :', e)

                    # Pour ajouter le lien du site imdb
                    if type(films) == str:
                        st.write(f" - [{films}]({url_imdb2})")
                    else:
                        st.write(f" - [{films}]({url_imdb2})")


    # choix du genre, de L'acteur et de la periode souhaitee
                                
    with st.form("form 2"):
        
        # Indication utilisateur
        st.subheader("2. Recherche par genres et acteur  sur une période voulu ")
        
        # Mise en place des choix utilisateurs
        col1, col2, col3 = st.columns(3)
        with col1 :
            acteurs = st.selectbox(label = "Acteurs : ", options = list_film_deroulante_acteurs)
        with col2 :
            genres = st.multiselect(label = "Genres : ", options = list_film_deroulante_genres)
            st.write("Vous avez selectionné", len(genres), 'genre(s)')
        with col3 :
            start_year, end_year = st.select_slider(label = "Période",
                                        options=liste_annees["startYear"],
                                        value=(1913, 2024))
            st.write("Vous avez selectionné la période allant de ", start_year, 'à', end_year)
        # le bouton submit
        submit_2 = st.form_submit_button("Rechercher")

        
        if submit_2 : 

            data_machine_learning_clean_year = pd.merge(liste_annees[liste_annees['startYear'].between(start_year, end_year)], data_machine_learning_clean, how = 'left', left_on = 'tconst', right_on = 'tconst')  

            
            data1 = data_machine_learning_clean_year[(data_machine_learning_clean_year[acteurs] == True)]
            
            #Création d'un dataframe filtré sur la periode et le genre 
            data2 = pd.DataFrame()
            for genre in genres : 
                data_genre= data_machine_learning_clean_year[ (data_machine_learning_clean_year[genre] == True) ]
                
                data2 = pd.concat([data2 , data_genre])


                
            # Classement  dans l'ordre decroissant de la colonne averageRating
            data1 = data1.sort_values(by ='averageRating' , ascending = False)
            data2 = data2.sort_values(by ='averageRating' , ascending = False)

            # TOP 5 des films  classés par note
            data_top5_acteur_choisi = data1.head(5)
            data_top4_genre_choisi = data2.head(4)

            # Récuperer uniquement les codes films tconst 
            tconst_acteur = data_top5_acteur_choisi.iloc[:,:]['tconst'].values
            tconst_genre = data_top4_genre_choisi.iloc[:,:]['tconst'].values

            # Récuperer uniquement les noms des films  
            suggestion_acteur = data_top5_acteur_choisi.iloc[:,:]['primaryTitle'].values
            suggestion_genre = data_top4_genre_choisi.iloc[:,:]['primaryTitle'].values

            # Prise en compte de l'absence d'acteur renseigné par l'utilisateur 
            if acteurs != "Choisis ton acteur préféré":

                # Indication utilisateur
                st.subheader("Si vous préférez {} je vous recommande  les 5 films suivants :".format(acteurs)  ) 

                # Création de colonnes
                col1 = st.columns(5)

                # PROPOSITION ACTEUR => Boucle simultanée sur les deux paramètres : suggestion et tconst 
                for films_acteur, code_film_acteur, colonnes_acteur in zip(suggestion_acteur,tconst_acteur, col1):

                    with colonnes_acteur:

                        url1 = f"{url_api}{code_film_acteur}{key_api}"  # on remplace tconst par code_film
                        url1_imdb2 = f"{url_imdb}{code_film_acteur}" 

                        try:
                            response = requests.get(url1)
                            #st.write(str(url))       
                            response.raise_for_status()
                            data = response.json()
                            url_image = data['Poster']
                            st.image(url_image, width=200)

                        except requests.exceptions.RequestException as e:
                            print('Une erreur s\'est produite lors de l\'appel à l\'API :', e)

                        # ajouter le lien du site imdb
                        if type(films_acteur) == str:
                            st.write(f" - [{films_acteur}]({url1_imdb2})")
                        else:
                            st.write(f" - [{films_acteur}]({url1_imdb2})")
                            
                            
            # Prise  en compte de l'absence de genres renseignés par l'utilisateur 
            if genres :

                st.subheader("En se basant sur les genres saisies, je vous recommande aussi  4 films  :" ) 

                # Création de colonnes
                col2 = st.columns(4) 

                # PROPOSITION GENRE ET ANNEES => Boucle simultanée sur les deux paramètres : suggestion et tconst 
                for films_genre, code_film_genre, colonnes_genre in zip(suggestion_genre,tconst_genre, col2):

                    with colonnes_genre:

                        url2 = f"{url_api}{code_film_genre}{key_api}"  # on remplace tconst par code_film
                        url2_imdb2 = f"{url_imdb}{code_film_genre}"   

                        try:
                            response = requests.get(url2)
                            #st.write(str(url))       
                            response.raise_for_status()
                            data = response.json()
                            url_image = data['Poster']
                            st.image(url_image, width=200)

                        except requests.exceptions.RequestException as e:
                            print('Une erreur est survenue lors de l\'appel à l\'API :', e)

                        # Pour ajouter le lien du site imdb
                        if type(films_genre) == str:
                            st.write(f" - [{films_genre}]({url2_imdb2})")
                        else:
                            st.write(f" - [{films_genre}]({url2_imdb2})")
        



    # SOUS-TITRE
    st.subheader("Bonne detente ! 🍿")
if page == 'App_modèle2':
    #Mise en page du fond de la page
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Définition des fonctions de chargement de fichier Lottie
    #def load_lottiefile(filepath: str):
        #with open(filepath, "r") as f:
            #return json.load(f)

    #def load_lottieurl(url: str):
        #r = requests.get(url)
        #if r.status_code != 200:
            #return None
        #return r.json()
    # Appel des deux fonctions
    #lottie_coding = load_lottiefile(r"C:\Users\user\Downloads\Projet ML\pages\image&animation\Animation1.json")  # replace link to local lottie file
    #lottie_hello = load_lottieurl("https://lottie.host/5c39a3f0-4225-4a25-ae7c-a9d6cd9ccbbd/m076uPOTI8.json")
    # Création d'une disposition personnalisée pour aligner l'animation et le titre sur la même ligne
    #col1, col2 = st.columns([1, 2])  # Divise l'espace en 3 colonnes, 1 pour l'animation et 2 pour le titre

    # Affichage de l'animation Lottie dans la première colonne
    #with col1:
        #st_lottie(
            #lottie_hello,
            #speed=1,
            #reverse=False,
            #loop=True,
            #quality="low",  # medium ; high
            #height=150,  # Hauteur de l'animation
            #width=150,  # Largeur de l'animation
        #)

    # Affichage du titre dans la deuxième colonne
    #with col2:
        #st.header('Bienvenue sur notre deuxième page de recommandation de films')

    # LISTE 
    list_film_deroulante_films = ["Saissisez le film que vous aimez"] + list(liste_films["primaryTitle"])

    # REQUETE API 

    site : "https://www.omdbapi.com/"
    #demande de key API : key = "26444a04"
    url_api = "http://www.omdbapi.com/?i="
    key_api = "&apikey=26444a04"
    url_imdb = 'https://www.imdb.com/title/' 

    # MACHINE LEARNING : 
    # Racinisation des mots dans un texte donné :
    # Avec l'algorithme Snowball
    sb = SnowballStemmer(language='english') # initialisation avec la langue anglaise
    def stem(text):
        L = []
        for i in text.split():
            L.append(sb.stem(i))
        string = " ".join(L)
        return string
    new_data['tags'] = new_data['tags'].apply(stem)
    #Création du modèle
    model2 = CountVectorizer(max_features=30000, stop_words='english')
    #Entrainer le modèle
    vector_matrix = model2.fit_transform(new_data['tags'])
    #Distance d'un film avec tous les autres films
    similarity = cosine_similarity(vector_matrix)
    #Recommandation
    with st.form("form 1"):
        
        # Indication utilisateur
        st.subheader("Recherche par titre de film")
        # Mise en place du choix utilisateurs
        films = st.selectbox("Films : ",list_film_deroulante_films)

        # Bouton submit
        submit = st.form_submit_button("Rechercher")

            
        # PROPOSITION DE L'ALGORITHME

        if submit : 

            def recommandation(movie):
                # Votre code pour obtenir les recommandations de films
                # Rechercher le film en fonction de primaryTitle, originalTitle ou frenchTitle
                movie_index = new_data[(new_data['primaryTitle'] == movie) | 
                                        (new_data['originalTitle'] == movie) | 
                                        (new_data['French_Title'] == movie)].index[0]
                # Extraire les distances du film donné avec tous les autres films
                distances = similarity[movie_index]
                # Trouver les 5 films les plus similaires excepté du film donné 
                movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
                # Créer une liste pour stocker les données des films recommandés
                recommendations_data = []

                for i in movies_list:
                    title = new_data.iloc[i[0]]['primaryTitle']
                    distance = i[1]
                    tconst = new_data.iloc[i[0]]['tconst']
                    
                    # Ajouter les données du film recommandé à la liste
                    recommendations_data.append({'tconst': tconst,'Film': title, 'Similarity': distance})

                # Créer un DataFrame à partir de la liste des données des films recommandés
                recommendations_df = pd.DataFrame(recommendations_data)
        
                return recommendations_df
            suggestion=recommandation(films)
            
            # Création de colonnes
            col1 = st.columns(4)

            # Boucle simultanée sur les deux paramètres : suggestion et tconst 
            for films, code_film, colonnes in zip(suggestion['Film'],suggestion['tconst'], col1):

                with colonnes:

                    url = f"{url_api}{code_film}{key_api}"  # on remplace tconst par code_film
                    url_imdb2 = f"{url_imdb}{code_film}"
                    try:
                        response = requests.get(url)
                        #st.write(str(url))       
                        response.raise_for_status()
                        data = response.json()
                        url_image = data['Poster']
                        st.image(url_image, width=200)

                    except requests.exceptions.RequestException as e:
                        print('Une erreur est survenue lors de l\'appel à l\'API :', e)

                    # Pour ajouter le lien du site imdb
                    if type(films) == str:
                        st.write(f" - [{films}]({url_imdb2})")
                    else:
                        st.write(f" - [{films}]({url_imdb2})")
    # SOUS-TITRE
    st.subheader("Bonne detente !🍿 ")
    

        

st.sidebar.write("Copyright © 2024 by Michel Sagesse Kolié")