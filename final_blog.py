import os
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated
import operator

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing the goal.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: str = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: str
    queries: List[str] = Field(default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: str = "1024x1024"
    quality: str = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add] 
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))



def router_node(state: State) -> dict:
    """Decides if the topic needs web research."""
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content="You are a routing module. Decide if web research is needed. Modes: closed_book, hybrid, open_book. If research is needed, provide queries."),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    
    return decision.model_dump()

def route_next(state: State) -> str:
    """Directs traffic based on the router's decision."""
    return "research" if state["needs_research"] else "orchestrator"

def research_node(state: State) -> dict:
    """Uses Tavily to search the web and extracts unique evidence."""
    queries = state.get("queries", []) or []
    raw_results = []
    tool = TavilySearchResults(max_results=6)

    for q in queries:
        raw_results.extend(tool.invoke({"query": q}))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content="Extract relevant evidence from raw search results. Include title, url, snippet."),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])

    dedup = {e.url: e for e in pack.evidence if e.url}
    return {"evidence": list(dedup.values())}

def orchestrator_node(state: State) -> dict:
    """Creates a detailed plan (outline) for the blog."""
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    
    # Format evidence simply
    top_evidence = evidence[:16]
    evidence_data = [e.model_dump() for e in top_evidence]
    
    plan = planner.invoke([
        SystemMessage(content="You are a senior technical writer. Create a 1-3 section outline (tasks) for this blog. Include target word counts and bullets."),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {state.get('mode')}\n"
            f"Evidence:\n{evidence_data}"
        ))
    ])
    return {"plan": plan}

def fanout(state: State):
    """Spawns parallel workers for every section in the plan."""
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]

def worker_node(payload: dict) -> dict:
    """Writes a single section of the blog."""
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])

    evidence_list = payload.get("evidence", [])
    top_20_evidence = evidence_list[:20]
    
    formatted_evidence = []
    for item in top_20_evidence:
        line = f"- {item['title']} | {item['url']}"
        formatted_evidence.append(line)
        
    evidence_text = "\n".join(formatted_evidence)
    # ---------------------------------------------

    section_md = llm.invoke([
        SystemMessage(content="Write ONE section of a technical blog in Markdown. Follow the goal and bullets exactly. Include code if required. Cite sources if using evidence."),
        HumanMessage(content=f"Blog Title: {plan.blog_title}\nSection: {task.title}\nGoal: {task.goal}\nBullets:\n{task.bullets}\nEvidence:\n{evidence_text}")
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}



def merge_content(state: State) -> dict:
    """Sorts the completed sections and merges them into one Markdown document."""
    

    sorted_sections = sorted(state["sections"], key=lambda x: x[0])
    
    section_texts = []
    for task_id, markdown_text in sorted_sections:
        section_texts.append(markdown_text)
        
    body_text = "\n\n".join(section_texts)
    
    blog_title = state["plan"].blog_title
    final_markdown = f"# {blog_title}\n\n{body_text}"
    
    return {"merged_md": final_markdown}

def decide_images(state: State) -> dict:
    """Decides where images should go and generates prompts for them."""
    planner = llm.with_structured_output(GlobalImagePlan)
    image_plan = planner.invoke([
        SystemMessage(content="Decide if images are needed. Max 3. Insert placeholders like [[IMAGE_1]]. Return ImageSpecs with prompts."),
        HumanMessage(content=f"Insert placeholders + propose image prompts.\n\n{state['merged_md']}")
    ])
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """Helper function to fetch raw image bytes from Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"])
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        parts = getattr(resp.candidates[0].content, "parts", [])

    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No image data found in response.")

def generate_and_place_images(state: State) -> dict:
    """Generates the actual images and replaces placeholders in the Markdown."""
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", [])

    if not image_specs:
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        out_path = images_dir / spec["filename"]
        
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                md = md.replace(spec["placeholder"], f"> **[IMAGE FAILED]** {e}\n")
                continue

        img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
        md = md.replace(spec["placeholder"], img_md)

    return {"final": md}

# Build Reducer Subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

def run_blog_maker(topic: str):
    """Executes the graph for a given topic."""
    result = app.invoke({
        "topic": topic,
        "mode": "", "needs_research": False, "queries": [], "evidence": [],
        "plan": None, "sections": [], "merged_md": "", "md_with_placeholders": "",
        "image_specs": [], "final": ""
    })
    
    if result.get("plan") and result.get("final"):
        filename = f"{result['plan'].blog_title}.md"
        Path(filename).write_text(result["final"], encoding="utf-8")
        print(f"Blog successfully saved to {filename}")
        
    return result