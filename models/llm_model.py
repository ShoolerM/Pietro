"""Model for managing LLM configuration and state."""

import os
import requests
from base.observable import Observable


class LLMModel(Observable):
    """Manages LLM configuration, available models, and generation state."""

    def __init__(self, base_url, temperature=0.7, api_key=""):
        super().__init__()
        # Set a dummy API key as environment variable since some servers don't require it
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = "not-needed"

        self._base_url = base_url
        self._temperature = temperature
        self._api_key = api_key or ""
        self._current_model = None
        self._available_models = []
        self._stop_generation = False

    @property
    def base_url(self):
        """Get LLM API base URL."""
        return self._base_url

    @base_url.setter
    def base_url(self, value):
        """Set LLM API base URL."""
        self._base_url = value
        self.notify_observers("base_url_changed", value)

    @property
    def api_key(self):
        """Get API key for inference server."""
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        """Set API key for inference server."""
        self._api_key = value or ""
        if self._api_key:
            os.environ["OPENAI_API_KEY"] = self._api_key
        else:
            os.environ["OPENAI_API_KEY"] = "not-needed"
        self.notify_observers("api_key_changed", self._api_key)

    @property
    def temperature(self):
        """Get temperature setting."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """Set temperature setting."""
        self._temperature = value
        self.notify_observers("temperature_changed", value)

    @property
    def current_model(self):
        """Get current model name."""
        return self._current_model

    @current_model.setter
    def current_model(self, value):
        """Set current model name."""
        self._current_model = value
        self.notify_observers("model_changed", value)

    @property
    def available_models(self):
        """Get list of available models."""
        return self._available_models

    @property
    def stop_generation(self):
        """Check if generation should stop."""
        return self._stop_generation

    def request_stop(self):
        """Request to stop current generation."""
        self._stop_generation = True
        self.notify_observers("stop_requested", None)

    def reset_stop_flag(self):
        """Reset the stop generation flag."""
        self._stop_generation = False
        self.notify_observers("stop_flag_reset", None)

    def fetch_available_models(self):
        """Fetch available models from the LLM API.

        Returns:
            tuple: (success: bool, models: list or error_message: str)
        """
        try:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = requests.get(
                f"{self._base_url}/models", timeout=5, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data and isinstance(data["data"], list):
                models = [model["id"] for model in data["data"] if "id" in model]
                if models:
                    self._available_models = models
                    self.notify_observers("models_fetched", models)
                    return True, models
                else:
                    return False, "No models available"
            else:
                return False, "No models available"
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return False, error_msg
