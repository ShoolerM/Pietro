"""Controller for settings dialogs and updates."""

from PyQt5 import QtWidgets


class SettingsController:
    """Handles settings dialogs and model updates."""

    def __init__(
        self,
        view,
        settings_model,
        llm_model,
        rag_model,
        save_model_profile_callback,
        refresh_models_callback,
    ):
        self.view = view
        self.settings_model = settings_model
        self.llm_model = llm_model
        self.rag_model = rag_model
        self._save_model_profile_callback = save_model_profile_callback
        self._refresh_models_callback = refresh_models_callback

    def on_summarization_prompt_requested(self):
        """Handle summarization prompt settings menu action."""
        saved, new_prompt = self.view.show_summarization_prompt_dialog(
            self.settings_model.summary_prompt_template
        )

        if saved and new_prompt is not None:
            success = self.settings_model.save_summary_prompt(new_prompt)
            if not success:
                self.view.show_warning(
                    "Save Error", "Failed to save summarization prompt"
                )

    def on_notes_prompt_requested(self):
        """Handle notes prompt settings menu action."""
        saved, new_prompt = self.view.show_notes_prompt_dialog(
            self.settings_model.notes_prompt_template
        )

        if saved and new_prompt is not None:
            success = self.settings_model.save_notes_prompt(new_prompt)
            if not success:
                self.view.show_warning("Save Error", "Failed to save notes prompt")

    def on_ask_prompt_requested(self):
        """Handle ask prompt settings menu action."""
        saved, new_prompt = self.view.show_ask_prompt_dialog(
            self.settings_model.ask_prompt_template
        )

        if saved and new_prompt is not None:
            success = self.settings_model.save_ask_prompt(new_prompt)
            if not success:
                self.view.show_warning("Save Error", "Failed to save ask prompt")

    def on_general_settings_requested(self):
        """Handle general settings menu action."""
        result = self.view.show_general_settings_dialog(
            self.settings_model.auto_notes, self.settings_model.render_markdown
        )

        if result.get("saved"):
            if result.get("auto_notes") is not None:
                self.settings_model.auto_notes = result["auto_notes"]
                self.view.append_logs(
                    f"✓ Auto Notes: {'enabled' if result['auto_notes'] else 'disabled'}"
                )

            if result.get("render_markdown") is not None:
                self.settings_model.render_markdown = result["render_markdown"]
                self.view.append_logs(
                    "✓ Render Markdown: "
                    f"{'enabled' if result['render_markdown'] else 'disabled'}"
                )

    def on_rag_settings_requested(self):
        """Handle RAG settings dialog request."""
        current_threshold_percent = self.rag_model.score_variance_threshold * 100.0
        self.view.show_rag_settings_dialog(
            current_max_chunks=self.rag_model.max_chunks,
            current_summary_chunk_size=self.rag_model.summary_chunk_size,
            current_score_threshold=current_threshold_percent,
            current_filename_boost_enabled=self.rag_model.filename_boost_enabled,
            current_max_filename_chunks=self.rag_model.max_filename_chunks,
            current_levenshtein_threshold=self.rag_model.levenshtein_threshold,
        )

    def on_model_settings_requested(self):
        """Handle model settings dialog request."""
        result = self.view.show_model_settings_dialog(
            current_context_limit=self.settings_model.context_limit
        )

        if result:
            self.settings_model.context_limit = result
            self.view.append_logs(f"✓ Context limit set to: {result} tokens")

    def on_inference_settings_requested(self):
        """Handle inference settings dialog request."""
        result = self.view.show_inference_settings_dialog(
            current_ip=self.settings_model.inference_ip,
            current_port=self.settings_model.inference_port,
            current_api_key=self.settings_model.api_key,
        )

        if not result:
            return

        ip, port, api_key = result

        test_url = f"http://{ip}:{port}/v1"
        self.view.append_logs(f"Testing connection to: {test_url}")

        old_url = self.llm_model.base_url
        old_key = self.llm_model.api_key
        self.llm_model.base_url = test_url
        self.llm_model.api_key = api_key

        success, result_data = self.llm_model.fetch_available_models()

        if success:
            self.settings_model.inference_ip = ip
            self.settings_model.inference_port = port
            self.settings_model.api_key = api_key
            self.settings_model.save_inference_settings()
            self.view.append_logs(f"✓ Inference server updated to: {ip}:{port}")
            self.view.append_logs(f"  Base URL: {self.settings_model.base_url}")

            self.llm_model.base_url = self.settings_model.base_url
            self.llm_model.api_key = self.settings_model.api_key

            if self._save_model_profile_callback:
                self._save_model_profile_callback()

            QtWidgets.QMessageBox.information(
                self.view,
                "Connection Successful",
                f"Successfully connected to inference server at {ip}:{port}\n\n"
                f"Found {len(result_data)} model(s).",
            )

            if self._refresh_models_callback:
                self._refresh_models_callback()
        else:
            self.llm_model.base_url = old_url
            self.llm_model.api_key = old_key

            QtWidgets.QMessageBox.critical(
                self.view,
                "Connection Failed",
                f"Could not connect to inference server at {ip}:{port}\n\n"
                f"Error: {result_data}\n\n"
                f"Please check:\n"
                f"• The IP address and port are correct\n"
                f"• The inference server is running\n"
                f"• There are no firewall issues",
            )
            self.view.append_logs(f"✗ Failed to connect to {ip}:{port}: {result_data}")
