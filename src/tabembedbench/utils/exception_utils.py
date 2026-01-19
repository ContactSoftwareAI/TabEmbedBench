class NotTaskCompatibleError(Exception):
    """Raised when the embedding model is not an end-to-end model."""

    def __init__(self, model, task_type):
        self.model = model
        self.task_type = task_type
        self.message = (
            f"The model {self.model} is not compatible with the task {self.task_type}."
        )
        super().__init__(self.message)


class NotEndToEndCompatibleError(Exception):
    """Raised when the embedding model is not an end-to-end model."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.message = (
            f"The benchmark {self.benchmark} is not compatible for end to end models."
        )
        super().__init__(self.message)
