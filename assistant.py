from crossref.restful import Works
from pyzotero import zotero
from os import listdir
import os
from os.path import isfile, join
import networkx as nx
from tqdm import tqdm
import math
import random
import numpy as np




class Recommendation_System(object):
    """docstring"""

    def __init__(self, library_id, api_key, k_best=10, depth=1, rand=0, groups=[]):

        self.zot = zotero.Zotero(library_id, "user", api_key)
        self.work = Works()
        self.k_best = k_best
        self.rand = rand
        self.depth = depth
        self.groups = groups

    def recommendation(self):

        if len(self.groups) == 0:
            self.collections = [collection["key"] for collection in self.zot.all_collections()]
        else:
            self.collections = [collection["key"] for collection in self.zot.all_collections() if collection["data"]["name"] in self.groups]
        for collection in tqdm(self.collections):

            # On récupère les article de la sous_collection
            test_articles = lambda art : not (art["DOI"] in ["", " ", "no"])
            article_collection = [article["data"] for article in self.zot.collection_items(collection) if "DOI" in article["data"].keys()]
            article_collection = [article["DOI"] for article in article_collection if test_articles(article)]

            if not os.path.exists(collection+".adjlist"):

                G = nx.DiGraph()

                G.add_nodes_from(article_collection)
                nx.write_adjlist(G,collection+".adjlist")

                articles_nouveaux = article_collection

            else:

                net = nx.read_adjlist(collection+".adjlist")
                test_presence = [article in net.nodes for article in article_collection]
                if np.all(test_presence):
                    continue
                else:
                    articles_nouveaux = [article for test,article in zip(article_collection,test_presence) if not test]


            network_link = self.citation_extraction(articles_nouveaux, 0, 2, rand=5)
            self.pg_score = self.network_update(collection, network_link)
            self.update_zotero(article_collection, collection)

        return None

    def update_zotero(self, articles, collection):

        # Récupérer la listes de tous les articles de zotero
        all_zotero = self.zot.items()
        all_zotero_doi = [item["data"]["DOI"] for item in all_zotero if "DOI" in item["data"]]

        article_to_add = [doi for doi in self.pg_score[:self.k_best] if not doi in all_zotero_doi]
        self.creer_article(article_to_add, collection)

        article_to_change = [doi for doi in self.pg_score[:self.k_best] if ((doi in all_zotero_doi) and (not doi in articles))]
        self.update_article(article_to_change,collection)

    def creer_article(self, doi_in, collection, verbose=1):

        for doi in doi_in:

            article = self.work.doi(doi)

            templ = self.zot.item_template("journalArticle")
            templ["title"] = article["title"][0]
            templ["collections"] = [collection]
            templ["DOI"] = doi


            resp = self.zot.create_items([templ])

            if verbose:
                print(article["titre"][0],"\n")

    def update_article(self, articles, collection):

        for article in articles:
            article["data"]["collections"] += [collection]
            self.zot.update_item(article)

    def network_update(self, collection, links):

        G = nx.read_adjlist(collection+".adjlist")
        G.add_edges_from(links)
        nx.write_adjlist(G, collection+".adjlist")

        pg = nx.pagerank(G)

        return sorted([clef for clef in pg.keys()],
                        key = lambda a : pg[a],
                        reverse=True)

    def liens_extraction(self, node_in):
        try:
            article = self.work.doi(node_in)
            out = [(node_in, art["DOI"]) for art in article["reference"] if "DOI" in art.keys()]
            return out
        except:
            return []

    def citation_extraction(self,nodes_doi,level=0,done=[]):
        # Fonction récursive qui
        rel = []
        if level < self.depth:
            for node in list(set(nodes_doi).difference(done)):

                rel += self.liens_extraction(node)
                nodes_more = list(set([b for (a,b) in rel]).difference(nodes_doi+done))


                if len(nodes_more) != 0:
                    if self.rand == 0:
                        rel_temp, done_temp = self.citation_extraction(nodes_more,level+1,self.depth,nodes_doi+done)
                        rel += rel_temp
                        done += done_temp
                        done += list(set([a for (a,b) in rel]))
                    else:
                        rand = math.floor(len(nodes_more)*self.rand) if self.rand < 1 else self.rand
                        nodes_more = random.sample(nodes_more,k=min(rand,len(nodes_more)))
                        rel_temp, done_temp = self.citation_extraction(nodes_more,level+1,self.depth,nodes_doi+done)
                        rel += rel_temp
                        done += done_temp
                        done += list(set([a for (a,b) in rel]))
                else:
                    continue
            if level == 0:
                return rel
            else:
                return rel, done
        else:
            return [], []

class Bibliotheque_Organisateur(object):
    """docstring forBibliotheque_Organisateur."""

    def __init__(self, arg):
        super(Bibliotheque_Organisateur, self).__init__()
        self.arg = arg

    def all_articles(self, ):
        """
            Méthode qui retourne tout les articles de Zotero
                * Articles
                * Relations
                * Titre
                * Abstract
        """
        pass

    def gen_graph(self, arg):
        """
            Méthode qui produit le graphe qui stocke tout ça
        """
        pass

# Variables pour l'authentification
lib_id = "9958949"
api_key = "pcIxJYdF8nIANd5nEUsXB2Ks"

"""
    Profondeur de la recherche
        * Plus c'est élevé plus on va aller chercher les citations de citations de citations etc...
        * Plus c'est élevé plus ça va prendre du temps (de façon exponentiel !!)
        * Plus c'est élevé plus on risque de s'éloigner du sujet initiale

        depth = 1 : on ne regarde que les citation des articles dans la base de donnée
        depth = 2 : on regarde les citations des articles cités par les articles de la base de données
        etc
"""
depth = 1

"""
    Si on fait des recherches profondes, le pourcentages d'articles cité par des citations qu'on veut évaluer
        * Pour améliorer les performances tout en se permettant d'augmenter la profondeur de la recherche
        * Plus c'est élevé plus la recherche sera fidèle à la vraie recherche

        * Si rand = 0, alors on prend tout les articles à chaque fois
        * Si rand < 1, c'est le % d'articles qu'on va prendre
            p.ex. si rand = 0.5 et qu'un article cité, cite lui même 100 articles, on va
            prendre 50 articles au hasard et les analyser eux aussi

        * Si rand > 1, on prend exactement le nombre indiqué, donc si rand = 5 on prend 5 articles à chaque fois
"""
rand = 10

"""
    Le nombre d'articles qu'on veut récupérer
        * Plus c'est élevé plus on risque de prendre des articles moins important
        * Si c'est trop bas, on prend le risque de ne prendre que trop peu d'articles
"""
k_best = 10

"""
    Les collections qu'on veut analyser
        * Si groups = []
            On analyse toutes les sections
        * Si groups = ["Déterminants","Verbes"]
            On analyse les sections indiquée, en l'occurence "Déterminants" et "Verbes"
"""
groups = []

assist = Recommendation_System(lib_id, api_key)
assist.recommendation()
