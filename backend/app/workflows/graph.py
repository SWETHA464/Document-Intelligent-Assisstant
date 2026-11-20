"""The analysis workflow.

    START -> route -> validate -> { qa | compare | extract | summarize } -> END
                 \\__ (error) ____________/

Built as a LangGraph `StateGraph` when `langgraph` is installed. If it is not, `SimpleWorkflow`
runs the very same node functions in the same order, so behaviour does not depend on it.
Every node catches expected `AppError`s and records them in `state["error"]`; the caller re-raises.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.core.context import AppContext
from app.core.errors import AppError
from app.schemas.analysis import AnalysisRequest
from app.services import comparison_service, extraction_service, qa_service, summary_service
from app.services.document_service import require_ready_documents
from app.services.field_presets import resolve_fields
from app.workflows.routing import OPERATIONS, infer_summary_type, route_operation
from app.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

Node = Callable[[WorkflowState], dict[str, Any]]


def _guard(name: str, fn: Callable[[WorkflowState], dict[str, Any]]) -> Node:
    """Wrap a node so expected errors land in state instead of escaping mid-graph."""

    def wrapper(state: WorkflowState) -> dict[str, Any]:
        try:
            update = fn(state)
        except AppError as exc:
            logger.info("Workflow step '%s' failed: %s", name, exc.message)
            return {"error": exc, "steps": [f"{name}:error"]}
        except Exception as exc:  # unexpected bug: report generically, keep details in the log
            logger.exception("Unexpected error in workflow step '%s'", name)
            error = AppError("An unexpected error occurred while analysing the documents.")
            error.status_code, error.code = 500, "internal_error"
            return {"error": error, "steps": [f"{name}:error"]}
        update.setdefault("steps", [name])
        return update

    return wrapper


def make_nodes(ctx: AppContext) -> dict[str, Node]:
    def route(state: WorkflowState) -> dict[str, Any]:
        return {"operation": route_operation(state["request"])}

    def validate(state: WorkflowState) -> dict[str, Any]:
        req: AnalysisRequest = state["request"]
        minimum = 2 if state["operation"] == "compare" else 1
        return {"documents": require_ready_documents(ctx, req.document_ids, minimum=minimum)}

    def qa(state: WorkflowState) -> dict[str, Any]:
        req = state["request"]
        result = qa_service.answer_question(ctx, req.question or "", state["documents"])
        return {"result": result.model_dump(mode="json")}

    def extract(state: WorkflowState) -> dict[str, Any]:
        req = state["request"]
        preset, fields = resolve_fields(preset=req.preset, custom=req.fields, text=req.question)
        result = extraction_service.extract_fields(ctx, state["documents"], fields, preset=preset)
        return {"result": result.model_dump(mode="json")}

    def compare(state: WorkflowState) -> dict[str, Any]:
        req = state["request"]
        preset, fields = resolve_fields(preset=req.preset, custom=req.fields, text=req.question)
        result = comparison_service.compare_documents(
            ctx, state["documents"], mode=req.compare_mode, fields=fields, preset=preset
        )
        return {"result": result.model_dump(mode="json")}

    def summarize(state: WorkflowState) -> dict[str, Any]:
        req = state["request"]
        summary_type = req.summary_type or infer_summary_type(req.question)
        result = summary_service.summarize_documents(ctx, state["documents"], summary_type)
        return {"result": result.model_dump(mode="json")}

    raw = {
        "route": route,
        "validate": validate,
        "qa": qa,
        "compare": compare,
        "extract": extract,
        "summarize": summarize,
    }
    return {name: _guard(name, fn) for name, fn in raw.items()}


# ------------------------------------------------------------- routing ----
def after_route(state: WorkflowState) -> str:
    return "end" if state.get("error") else "validate"


def after_validate(state: WorkflowState) -> str:
    if state.get("error"):
        return "end"
    op = state.get("operation", "")
    return op if op in OPERATIONS else "end"


# ----------------------------------------------------------- runners ----
class SimpleWorkflow:
    """Dependency-free runner with the same semantics as the LangGraph graph."""

    def __init__(self, nodes: dict[str, Node]) -> None:
        self.nodes = nodes

    @staticmethod
    def _apply(state: WorkflowState, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if key == "steps":
                state["steps"] = [*state.get("steps", []), *value]
            else:
                state[key] = value  # type: ignore[literal-required]

    def invoke(self, state: WorkflowState) -> WorkflowState:
        state = dict(state)  # type: ignore[assignment]
        state.setdefault("steps", [])
        self._apply(state, self.nodes["route"](state))
        if after_route(state) == "end":
            return state
        self._apply(state, self.nodes["validate"](state))
        nxt = after_validate(state)
        if nxt != "end":
            self._apply(state, self.nodes[nxt](state))
        return state


def build_workflow(ctx: AppContext, *, use_langgraph: bool | None = None):
    """Return an object with `.invoke(state) -> state`.

    use_langgraph: None = use LangGraph if installed; False = force the plain runner.
    """
    nodes = make_nodes(ctx)
    if use_langgraph is not False:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            if use_langgraph:
                raise
            logger.info("langgraph is not installed; using the built-in workflow runner.")
            return SimpleWorkflow(nodes)

        graph = StateGraph(WorkflowState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", after_route, {"validate": "validate", "end": END})
        graph.add_conditional_edges(
            "validate", after_validate, {**{op: op for op in OPERATIONS}, "end": END}
        )
        for op in OPERATIONS:
            graph.add_edge(op, END)
        return graph.compile()
    return SimpleWorkflow(nodes)
