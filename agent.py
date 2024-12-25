from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

from tools import download_pdf, fetch_arxiv_papers


class Agent:
    def __init__(self, index, llm_model):
        self.index = index
        self.llm_model = llm_model
        self.build_query_engine()
        self.build_rag_tool()
        self.build_pdf_download_tool()
        self.build_fetch_arxiv_tool()
        self.build_agent()

    def build_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            llm_model=self.llm_model, similarity_top_k=5
        )

    def build_rag_tool(self):
        self.rag_tool = QueryEngineTool.from_defaults(
            self.query_engine,
            name="research_paper_query_engine_tool",
            description="A RAG engine with recent research papers.",
        )

    def build_pdf_download_tool(self):
        self.download_pdf_tool = FunctionTool.from_defaults(
            download_pdf,
            name="download_pdf_file_tool",
            description="python function that downloads a pdf file by link",
        )

    def build_fetch_arxiv_tool(self):
        self.fetch_arxiv_tool = FunctionTool.from_defaults(
            fetch_arxiv_papers,
            name="fetch_from_arxiv",
            description="download the {max_results} recent papers regarding the topic {title} from arXiv",
        )

    def build_agent(self):
        self.agent = ReActAgent.from_tools(
            [self.download_pdf_tool, self.rag_tool, self.fetch_arxiv_tool],
            llm=self.llm_model,
            verbose=True,
        )

    def chat(self, message: str):
        return self.agent.chat(message)
