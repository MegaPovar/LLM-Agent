from . import AgentBase

class EvalAgent(AgentBase):
    name = "eval"
    def run(self, ctx):
        ctx["brief"] = "Evaluation complete."
        self.save_context(ctx)
        return ctx
