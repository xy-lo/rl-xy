"""Explicit wafer mesh geometry for architecture-dependent planning.

This module is a topology foundation only; the legacy runner still assumes H0.
It does not claim that a generated layout is executable by the native backend.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MeshGeometry:
    columns: int
    rows: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.columns, self.rows)):
            raise ValueError('mesh dimensions must be positive integers')

    @property
    def die_count(self):
        return self.columns * self.rows

    def coordinate(self, die):
        if type(die) is not int or not 0 <= die < self.die_count:
            raise ValueError('die outside mesh')
        return divmod(die, self.columns)

    def die_at(self, row, column):
        if any(type(v) is not int for v in (row,column)) or not (0 <= row < self.rows and 0 <= column < self.columns):
            raise ValueError('coordinate outside mesh')
        return row * self.columns + column

    def neighbors(self, die):
        row,col=self.coordinate(die)
        return tuple(self.die_at(y,x) for y,x in ((row-1,col),(row+1,col),(row,col-1),(row,col+1))
                     if 0 <= y < self.rows and 0 <= x < self.columns)

    def distance(self, a, b):
        ay,ax=self.coordinate(a);by,bx=self.coordinate(b)
        return abs(ay-by)+abs(ax-bx)

    def adjacent_pairs(self):
        """One possible complete matching except one leftover for odd die counts.

        Horizontal pairs use the legacy H0 numbering; remaining odd-column
        nodes are paired vertically. Returned IDs must never be converted to
        die IDs by multiplying worker IDs by two on odd-column meshes.
        """
        w,h=self.columns,self.rows
        pairs=[(self.die_at(y,x),self.die_at(y,x+1))
               for y in range(h) for x in range(0,w-1,2)]
        if w%2:
            pairs.extend((self.die_at(y,w-1),self.die_at(y+1,w-1))
                         for y in range(0,h-1,2))
        used={die for pair in pairs for die in pair}
        unused=tuple(die for die in range(self.die_count) if die not in used)
        if len(used)!=2*len(pairs) or len(unused)!=self.die_count%2:
            raise AssertionError('invalid adjacent matching')
        return tuple(pairs),unused

    def validate_partition(self, groups, *, allow_unassigned=False):
        """Validate disjoint, connected physical groups; no performance claim."""
        seen=set()
        for group in groups:
            group=tuple(group)
            if not group or len(set(group))!=len(group):
                raise ValueError('empty or duplicate group')
            for die in group:self.coordinate(die)
            nodes=set(group)
            if seen & nodes:raise ValueError('overlapping groups')
            reached={group[0]};frontier=[group[0]]
            while frontier:
                for neighbor in self.neighbors(frontier.pop()):
                    if neighbor in nodes and neighbor not in reached:
                        reached.add(neighbor);frontier.append(neighbor)
            if reached!=nodes:raise ValueError('disconnected group')
            seen.update(nodes)
        unused=tuple(sorted(set(range(self.die_count))-seen))
        if unused and not allow_unassigned:raise ValueError('incomplete coverage')
        return unused


@dataclass(frozen=True)
class PairMapping:
    """Physical ranks for canonical two-lane optimizer/storage owners.

    Worker numbering is logical; only this mapping converts it to die IDs.
    Unpaired nodes are explicitly excluded from these workers and can be used
    only by a separately implemented single-die execution path.
    """
    mesh: MeshGeometry
    pairs: tuple
    unpaired: tuple

    @classmethod
    def adjacent(cls, mesh):
        pairs,unpaired=mesh.adjacent_pairs()
        return cls(mesh,pairs,unpaired)

    @classmethod
    def with_idle_tail_pairs(cls, mesh, count):
        if type(count) is not int or count < 0:
            raise ValueError('invalid idle pair count')
        pairs,unused=mesh.adjacent_pairs()
        if count>=len(pairs):raise ValueError('no active owner pairs')
        if not count:return cls(mesh,pairs,unused)
        retired=pairs[-count:]
        return cls(mesh,pairs[:-count],tuple(sorted(unused+tuple(d for p in retired for d in p))))

    def __post_init__(self):
        if any(len(pair)!=2 or self.mesh.distance(*pair)!=1 for pair in self.pairs):
            raise ValueError('each worker requires two adjacent dies')
        actual=self.mesh.validate_partition(self.pairs,allow_unassigned=True)
        if tuple(self.unpaired)!=actual:raise ValueError('unpaired die list mismatch')

    def inference_ranks(self, workers):
        workers=tuple(workers)
        if len(workers)!=len(set(workers)) or any(type(w) is not int or w<0 or w>=len(self.pairs) for w in workers):
            raise ValueError('invalid worker list')
        return tuple(self.pairs[w][lane] for lane in range(2) for w in workers)

    def owner_lane(self, die):
        self.mesh.coordinate(die)
        for pair in self.pairs:
            if die in pair:return pair.index(die)
        raise ValueError('die is not a paired owner')

    def adjacent_workers(self, a, b):
        if type(a) is not int or type(b) is not int or not (0<=a<len(self.pairs) and 0<=b<len(self.pairs)):
            raise ValueError('worker outside mapping')
        return any(self.mesh.distance(x,y)==1 for x in self.pairs[a] for y in self.pairs[b])
