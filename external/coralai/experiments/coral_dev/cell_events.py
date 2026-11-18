"""Optional observation of genome transitions at simulator phase boundaries."""


class CellEvents:
    phases = ("colonization", "resource_death", "chunk_death", "radiation", "cleanup", "cull")

    def __init__(self, substrate):
        self.substrate = substrate
        self.genome_index = int(substrate.ti_indices[None].genome)

    def begin(self, update):
        self.update = update
        self.previous = self.substrate.mem[0, self.genome_index].clone()
        self.initial = self.previous >= 0
        self.values = {}

    def observe(self, phase):
        current = self.substrate.mem[0, self.genome_index]
        before, after = self.previous >= 0, current >= 0
        self.values[phase + "_gains"] = int((~before & after).sum())
        self.values[phase + "_losses"] = int((before & ~after).sum())
        self.values[phase + "_replacements"] = int((before & after & (self.previous != current)).sum())
        self.previous = current.clone()

    def finish(self):
        final = self.previous >= 0
        result = {"completed_update": self.update, "occupied_before": int(self.initial.sum()),
                  "occupied_after": int(final.sum()), "net_gains": int((~self.initial & final).sum()),
                  "net_losses": int((self.initial & ~final).sum())}
        result.update({p + "_" + k: self.values.get(p + "_" + k, 0)
                       for p in self.phases for k in ("gains", "losses", "replacements")})
        assert result["occupied_after"] - result["occupied_before"] == sum(
            result[p + "_gains"] - result[p + "_losses"] for p in self.phases)
        return result
