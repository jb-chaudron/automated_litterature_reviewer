from semanticscholar import SemanticScholar
from S2search import S2paperAPI
import spacy
import spacy_fastlang
import networkx as nx
from tqdm import tqdm 

class ArtificialLibraire():
    def __init__(self) -> None:
        self.search_engine = S2paperAPI()
        self.reference_searcher = SemanticScholar()
        self.litterature_graph = nx.DiGraph()

    def doc_from_research(self, request, field=["Computer Science"], nb_paper=40):
        self.search_engine.get(request,
                               n=nb_paper,
                               fieldsOfStudy=field)
        self.papers_id = [id for id in self.search_engine.all.paperId]

    def doc_from_zotero(self, DOI_in):
        self.papers_id = [self.reference_searcher.get_paper(doi).paperId for doi in DOI_in]

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
        