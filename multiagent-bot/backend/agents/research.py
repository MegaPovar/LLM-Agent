from . import AgentBase

class ResearchAgent(AgentBase):
    name = "research"

    def run(self, ctx: dict) -> dict:
        # пока просто заглушка
        ctx["findings"]["research"] = {"note": "Research step placeholder"}
        ctx["brief"] = "Research step complete."
        self.save_context(ctx)
        return ctx