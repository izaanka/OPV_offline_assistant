"""Online context proxy module — legacy compatibility wrapper for modules_registry."""

from typing import Optional
import modules_registry

def get_online_context(user_input: str) -> Optional[str]:
    """Compatibility wrapper redirecting to the dynamic modules system."""
    return modules_registry.get_direct_context(user_input)
