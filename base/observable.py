"""Observable base class for implementing observer pattern."""


class Observable:
    """Base class for models that can be observed for changes."""
    
    def __init__(self):
        self._observers = []
    
    def add_observer(self, observer):
        """Add an observer callback that will be notified of changes.
        
        Args:
            observer: A callable that accepts (event_type, data) parameters
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer):
        """Remove an observer callback.
        
        Args:
            observer: The observer callback to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_observers(self, event_type, data=None):
        """Notify all observers of a change.
        
        Args:
            event_type: String describing the type of event
            data: Optional data associated with the event
        """
        for observer in self._observers:
            try:
                observer(event_type, data)
            except Exception as e:
                print(f"Error notifying observer: {e}")
