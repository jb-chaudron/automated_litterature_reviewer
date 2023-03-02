from semanticscholar import SemanticScholar
from S2search import S2paperAPI
import spacy
import spacy_fastlang
import networkx as nx
from tqdm import tqdm 
from math import floor
import numpy as np
import networkx.algorithms.community as nx_comm
from scipy.spatial.distance import squareform, pdist
from bertopic import BERTopic


class ArtificialLibraire():
    def __init__(self) -> None:
        self.search_engine = S2paperAPI()
        self.reference_searcher = SemanticScholar()
        self.litterature_graph = nx.DiGraph()
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.add_pipe("language_detector")
    """
        Methods to get the list of paper initializing the all process
            Either by looking with a search engine
            Either by giving the list of paper as input
    """
    def doc_from_research(self, request, field=["Computer Science"], nb_paper=40):
        self.search_engine.get(request,
                               n=nb_paper,
                               fieldsOfStudy=field)
        self.papers_id = [id for id in self.search_engine.all.paperId]

    def doc_from_zotero(self, DOI_in):
        self.papers_id = [self.reference_searcher.get_paper(doi).paperId for doi in DOI_in]

    """
        Methods to get the graph of nearest neighbors in the citation space
    """
    def get_graph(self, core=False):
        get_ref = lambda a : [(a["paperId"],b["paperId"]) for b in a.references if ((not b["paperId"] is None) and (not b is None))]
        get_citation = lambda a : [(b["paperId"],a["paperId"]) for b in a.citations if ((not b["paperId"] is None) and (not b is None))]

        self.litterature_graph.add_nodes_from([(node, {"processed" : False,
                                                       "core" : core}) for node in self.papers_id])
        
        core_attributes = nx.get_node_attributes(self.litterature_graph, "core")
        processed_attributes = nx.get_node_attributes(self.litterature_graph, "processed")

        nodes_attributes = [(node, is_core, processed_attributes[node]) for (node, is_core) in core_attributes.items()]
        node_to_process = [nom for (nom, is_core, is_processed) in nodes_attributes if (is_core and not is_processed)]

        for node_name in tqdm(node_to_process):
            if self.litterature_graph.nodes[node_name]["processed"]:
                continue
            else :

                paper = self.reference_searcher(node_name)
                paper_id = paper.paperId

                get_links_data = lambda paper_in : [(linked_paper.paperId,linked_paper.title,linked_paper.abstract) for linked_paper in paper_in]
                refs = get_links_data(paper.references)
                citations = get_links_data(paper.citations)

                link_reference = [(paper_id, reference[0]) for reference in refs if not (reference[0] is None)]
                link_citation = [(citation[0], paper_id) for citation in citations if not (citation[0] is None)]

                self.litterature_graph.add_edges_from(link_reference+link_citation)

                ids = [paper_id]+[reference[0] for reference in refs]+[citation[0] for citation in citations]
                titres = [paper.title]+[reference[1] for reference in refs]+[citation[1] for citation in citations]
                abstracts = [paper.abstract]+[reference[2] for reference in refs]+[citation[2] for citation in citations]

                attributes = {node : {"title" : titre,
                                      "abstract" : abstract,
                                      "processed" : True if node == paper_id else self.check_attribute(node,"processed"),
                                      "core" : core if node == paper_id else self.check_attribute(node, "core")}
                                      for (node, titre, abstract) in zip(ids, titres, abstracts)}
                
                nx.set_node_attributes(self.litterature_graph, attributes)

    def check_attribute(self, node_name, attribute):
        if not node_name in self.litterature_graph.nodes:
            return False
        else:
            return self.litterature_graph.nodes[node_name][attribute]
        
    """
        Methods to increase the set of interesting papers
    """
    def update_core_paper(self, thresh_in_degree, thresh_out_degree):
        core_attribute = nx.get_node_attributes(self.litterature_graph,"core")
        core_papers = [node for (node, is_core) in core_attribute.items() if is_core]

        thresh_in_degree = self.transform_threshold(thresh_in_degree, core_papers)
        thresh_out_degree = self.transform_threshold(thresh_out_degree, core_papers)

        # Papers highly cited by paper that we consider as interesting
        degree = [(node, degree) for (node, degree) in self.litterature_graph.in_degree if degree > thresh_in_degree]
        for node, _ in degree:
            nx.set_node_attributes(self.litterature_graph,{node : {"core" : True}})
        
        # Papers citing interesting papers
        un, count = self.get_interesting_nodes()
        interesting_nodes = un[count>thresh_out_degree]
        for node in interesting_nodes:
            nx.set_node_attributes(self.litterature_graph,{node : {"core" : True}})

    def transform_threshold(self, threshold, core_papers):
        if threshold < 0:
            raise Exception("threshold inferior to Zero")
        elif threshold < 1:
            threshold = max(1,floor(len(core_papers)*threshold))
        else:
            threshold = threshold

        return threshold
    
    def get_interesting_nodes(self):
        core_attributes = nx.get_node_attributes(self.litterature_graph, "core")
        core_nodes = [node for (node, is_core) in core_attributes.items() if is_core]

        nodes_out = [citing_paper for node in core_nodes 
                                  for citing_paper in self.litterature_graph.predecessors(node)
                                  if not self.litterature_graph.nodes[citing_paper]["core"]]

        return np.unique(nodes_out, return_counts=True)

    """
        Methods to update graph using NLP informations
    """

    def abstract2doc(self):
        nodes = [node for node in self.litterature_graph.nodes if self.litterature_graph.nodes[node]["core"]]
        abstracts = [self.litterature_graph.nodes[node]["abstract"] for node in nodes]
        
        self.docs = [doc for doc in tqdm(self.nlp.pipe([abstract if abstract != None else " " for abstract in abstracts]), total = len(abstracts))]
        self.docs = {node : doc for (node,doc) in zip(nodes, self.docs) if ((len(doc) > 10) and (doc._.language == "en"))}

    def add_weight_to_graph(self):

        similarity_weight = lambda doc_a, doc_b : doc_a.similarity(doc_b)
        weight_edges = {(node_out, node_in) : 
                                {"weight" : similarity_weight(self.docs[node_out], self.docs[node_in])} 
                                        for (node_out, node_in) in self.litterature_graph.edges}
        nx.set_edge_attributes(self.litterature_graph, weight_edges)


    def get_communities(self, resolution=1):
        self.communities = nx_comm.louvain_communities(self.litterature_graph,"weight", resolution=1)
    

    """
        Methods to perform topic analysis
    """

    def bert_topic(self):
        nlp_bertopic = spacy.load("en_core_web_lg",
                                  exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
        
        topic_model = BERTopic(embedding_model=nlp_bertopic)

        abstract_attribute = nx.get_node_attributes(self.litterature_graph,"abstract")

        topics, probs = topic_model.fit_transform([abstract for (node,abstract) in 
                                                        abstract_attribute.items()])
        hierarchical_topics = topic_model.hierarchical_topics([abstract for (node,abstract) in 
                                                                                abstract_attribute.items()])
