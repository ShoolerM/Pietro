"""BAML Controller

Generates structured story outlines using a single direct LLM call via baml-py.
BAML sends the user's planning request (plus context) to the model and returns
a typed list of OutlinePlotPoints — no second-pass parsing needed and no
response_format or function-calling support required from the provider.
Works with any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, etc.)

The BAML schema lives in models/baml_src/outline.baml.
"""

import os
import types
from pathlib import Path
from typing import Callable, Optional

import models.planning_model as _planning_model_module
from baml_py import BamlRuntime, ClientRegistry, FunctionResult, FunctionResultStream

# cast_to() requires real PyModule objects for enum_module and cls_module.
# _planning_model_module already contains OutlinePlotPoint so BAML can resolve
# class names from the schema via attribute lookup.
# _empty_module satisfies the enum_module argument (no enums in our schema).
_empty_module = types.ModuleType("_baml_empty")

# Absolute path to models/baml_src/ which contains outline.baml
_BAML_SRC_DIR: Path = Path(__file__).parent.parent / "models" / "baml_src"


class BamlController:
    """Controller for direct BAML-based story outline generation.

    Sends the user's story planning request directly to the LLM via BAML's
    GenerateOutline function and returns a typed list of OutlinePlotPoints.
    A single LLM call replaces the old two-pass stream-then-parse approach.
    Works without response_format or function-calling support from the provider.

    Usage:
        baml_ctrl = BamlController(view)
        plot_points = baml_ctrl.generate_outline(
            context="...",
            user_request="Create a 5-act outline for my story",
            base_url="http://localhost:1234/v1",
            model="my-model",
            api_key="",
        )
    """

    def __init__(self, view) -> None:
        """Initialize the BAML runtime with the outline schema from models/baml_src/outline.baml.

        Args:
            view: Main application view, used for append_logs() and show_warning().
        """
        self._view = view
        self._runtime = None

        # Load the BAML schema from models/baml_src/outline.baml via BamlRuntime.
        try:
            self._runtime = BamlRuntime.from_directory(
                str(_BAML_SRC_DIR),
                dict(os.environ),
            )
        except Exception as exc:
            msg = f"BAML failed to initialize: {exc}"
            self._view.append_logs(f"❌ {msg}\n")
            self._view.show_warning(
                "Outline Parsing Unavailable",
                f"{msg}\n\nStructured outline parsing is disabled for this session.",
            )

    @property
    def is_available(self) -> bool:
        """Return True if the BAML runtime initialized successfully."""
        return self._runtime is not None

    def _build_client_registry(self, base_url: str, model: str, api_key: str) -> ClientRegistry:
        """Build a ClientRegistry pointing at the user's configured LLM provider."""
        client_registry = ClientRegistry()
        client_options: dict = {"base_url": base_url, "model": model}
        # Only attach api_key when non-empty to avoid sending an Authorization
        # header to local servers that reject it.
        if api_key:
            client_options["api_key"] = api_key
        client_registry.add_llm_client(
            name="PietroLLM",
            provider="openai-generic",
            options=client_options,
        )
        client_registry.set_primary("PietroLLM")
        return client_registry

    def plan(
        self,
        context: str,
        conversation_history: str,
        user_message: str,
        base_url: str,
        model: str,
        api_key: str,
        on_partial: Optional[Callable] = None,
    ) -> Optional[object]:
        """Agentic planning turn via the BAML Plan() function.

        The LLM decides whether to have a conversation (action='chat') or
        generate/update an outline (action='outline'). Returns a
        PlanningResponse, or None on failure.

        on_partial is called whenever the parsed partial PlanningResponse
        meaningfully changes: each new text delta for 'chat', or each new
        plot point for 'outline'.

        Args:
            context: Dynamic context additions (story, notes, RAG, etc.).
            conversation_history: Prior turns as 'Human: ...\nAssistant: ...' text.
            user_message: The user's current message.
            base_url: LLM API base URL.
            model: Model identifier string.
            api_key: API key (empty string for local servers).
            on_partial: Optional callback receiving each partial PlanningResponse.

        Returns:
            PlanningResponse instance, or None on failure.
        """
        if self._runtime is None:
            return None

        try:
            client_registry = self._build_client_registry(base_url, model, api_key)
            ctx = self._runtime.create_context_manager()

            _prev_msg_len: list = [0]
            _prev_pts_count: list = [0]

            def _on_event(partial_result: FunctionResult) -> None:
                if on_partial is None:
                    return
                try:
                    partial_typed = partial_result.cast_to(
                        _empty_module,
                        _planning_model_module,
                        _planning_model_module,
                        True,
                        self._runtime,
                    )
                    if partial_typed is None:
                        return
                    action = getattr(partial_typed, "action", None)
                    if action == "chat":
                        msg = getattr(partial_typed, "message", None) or ""
                        if len(msg) > _prev_msg_len[0]:
                            _prev_msg_len[0] = len(msg)
                            on_partial(partial_typed)
                    elif action == "outline":
                        pts = getattr(partial_typed, "plot_points", None) or []
                        if len(pts) > _prev_pts_count[0]:
                            _prev_pts_count[0] = len(pts)
                            on_partial(partial_typed)
                except Exception:
                    pass

            stream: FunctionResultStream = self._runtime.stream_function_sync(
                "Plan",
                {
                    "context": context,
                    "conversation_history": conversation_history,
                    "user_message": user_message,
                },
                _on_event,
                ctx,
                None,
                client_registry,
                [],
                dict(os.environ),
                {},
            )
            result_raw: FunctionResult = stream.done(ctx)

            if not result_raw.is_ok():
                self._view.append_logs("❌ [BAML] Plan returned a failed result.\n")
                return None

            typed_result = result_raw.cast_to(
                _empty_module,
                _planning_model_module,
                _planning_model_module,
                False,
                self._runtime,
            )

            if typed_result is None:
                self._view.append_logs("❌ [BAML] Plan returned no result.\n")
                return None

            return typed_result

        except Exception as exc:
            self._view.append_logs(f"❌ [BAML] Plan failed: {exc}\n")
            return None

    def maybe_update_outline(
        self,
        current_outline: str,
        user_message: str,
        assistant_reply: str,
        base_url: str,
        model: str,
        api_key: str,
    ) -> Optional[list]:
        """Check whether a conversational exchange warrants an outline update.

        Calls MaybeUpdateOutline with the current outline and the just-completed
        exchange.  Returns an updated list of OutlinePlotPoints if the outline
        should change, or None if no update is needed (including on failure).

        Args:
            current_outline: The current outline checklist string.
            user_message: The user's message from this turn.
            assistant_reply: The assistant's streamed reply text.
            base_url: LLM API base URL.
            model: Model identifier string.
            api_key: API key (empty string for local servers).

        Returns:
            List[OutlinePlotPoint] if the outline should be updated, else None.
        """
        if self._runtime is None:
            return None

        try:
            client_registry = self._build_client_registry(base_url, model, api_key)
            ctx = self._runtime.create_context_manager()

            result_raw: FunctionResult = self._runtime.call_function_sync(
                "MaybeUpdateOutline",
                {
                    "current_outline": current_outline,
                    "user_message": user_message,
                    "assistant_reply": assistant_reply,
                },
                ctx,
                None,
                client_registry,
                [],
                dict(os.environ),
                {},
            )

            if not result_raw.is_ok():
                return None

            typed_result = result_raw.cast_to(
                _empty_module,
                _planning_model_module,
                _planning_model_module,
                False,
                self._runtime,
            )

            # null → None (no update needed); non-empty list → updated points
            if typed_result is None:
                return None
            if isinstance(typed_result, list) and typed_result:
                return typed_result
            return None

        except Exception as exc:
            self._view.append_logs(f"⚠️ [BAML] MaybeUpdateOutline failed: {exc}\n")
            return None
