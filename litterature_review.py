from semanticscholar import SemanticScholar
from S2search import S2paperAPI
import spacy
import spacy_fastlang


class ArtificialLibraire():
    def __init__(self) -> None:
        self.search_engine = S2paperAPI()
        self.reference_searcher = SemanticScholar()

    def doc_from_research(self, request, field=["Computer Science"], nb_paper=40):
        self.search_engine.get(request,
                               n=nb_paper,
                               fieldsOfStudy=field)
        self.papers_id = [id for id in self.search_engine.all.paperId]

    def doc_from_zotero(self, DOI_in):
        self.papers_id = [self.reference_searcher.get_paper(doi).paperId for doi in DOI_in]

    