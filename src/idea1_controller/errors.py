"""Explicit integration failures; the framework never substitutes a strategy."""
class ControllerError(RuntimeError):
    pass

class LifecycleError(ControllerError):
    pass

class ContractViolation(ControllerError):
    pass

class UnsupportedPlan(ControllerError):
    pass

class NotReady(ControllerError):
    pass

class ExecutionFault(ControllerError):
    pass
