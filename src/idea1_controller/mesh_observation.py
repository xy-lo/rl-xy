"""Observed directed queues and committed messages, not future trace lengths.

The native backend chooses a single XY/YX route per admitted logical message.
Estimates below rank bounded alternatives; they are not completion guarantees.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MeshObservation:
    tick_ps: int
    width: int
    height: int
    edges: tuple
    policy: str
    native_ticks_per_ns: float

    @classmethod
    def parse(cls, raw, tick_ps):
        fields=['src','dst','announced_not_started_bytes','queued_here_bytes','actual_queue_wait_ns','committed_service_ns']
        if not isinstance(raw,dict) or raw.get('schema')!='rlxy.mesh-observed/2' or raw.get('edge_fields')!=fields:
            raise ValueError('unsupported native mesh telemetry')
        scale=raw.get('native_ticks_per_ns')
        if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<1:
            raise ValueError('native clock scale missing or invalid')
        width,height=raw['width'],raw['height']
        if type(width) is not int or type(height) is not int or min(width,height)<2 or type(tick_ps) is not int or tick_ps<0:
            raise ValueError('invalid mesh dimensions or observation time')
        if raw['policy'] not in ('xy','adaptive_xy_yx'):raise ValueError('invalid route policy')
        seen=set();rows=[]
        for row in raw['edges']:
            if len(row)!=6:raise ValueError('malformed directed-edge counters')
            src,dst,announced,queued,wait,work=row
            if type(src) is not int or type(dst) is not int or not 0<=src<width*height or not 0<=dst<width*height:
                raise ValueError('invalid mesh endpoint')
            if abs(src//width-dst//width)+abs(src%width-dst%width)!=1:
                raise ValueError('nonadjacent physical link')
            if (src,dst) in seen:raise ValueError('duplicate directed counter')
            if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<0 for x in row[2:]):
                raise ValueError('invalid counter')
            if queued>announced:raise ValueError('queued bytes exceed declared unstarted bytes')
            seen.add((src,dst));rows.append(tuple(row))
        return cls(tick_ps,width,height,tuple(sorted(rows)),raw['policy'],float(scale))

    def fresh(self,tick_ps,max_age_ps=0):
        return self.tick_ps<=tick_ps<=self.tick_ps+max_age_ps

    def paths(self,src,dst):
        if any(type(x) is not int or not 0<=x<self.width*self.height for x in (src,dst)):
            raise ValueError('endpoint outside physical mesh')
        result=[]
        for xy in (True,False):
            current=src;path=[src]
            for horizontal in ((True,False) if xy else (False,True)):
                if horizontal:
                    while current%self.width!=dst%self.width:
                        current+=1 if current%self.width<dst%self.width else -1;path.append(current)
                else:
                    while current//self.width!=dst//self.width:
                        current+=self.width if current//self.width<dst//self.width else -self.width;path.append(current)
            if tuple(path) not in result:result.append(tuple(path))
        return tuple(result)

    def transfer(self,src,dst,size,bandwidth_B_ns,latency_ns,packet_bytes=65536):
        if type(size) is not int or size<0 or not math.isfinite(bandwidth_B_ns) or bandwidth_B_ns<=0 or latency_ns<0 or not math.isfinite(latency_ns):
            raise ValueError('invalid transport contract')
        candidates=self.paths(src,dst)
        if src==dst or size==0:return dict(service_ns=0.,queue_penalty_ns=0.,path=candidates[0],scope='no network payload')
        work={(r[0],r[1]):float(r[5]) for r in self.edges}
        def score(path):return sum(work.get(edge,0.) for edge in zip(path,path[1:]))
        path=candidates[0];base=score(path)
        if self.policy=='adaptive_xy_yx' and len(candidates)>1:
            other=score(candidates[1])
            if base-other>max(1.,.05*base):path=candidates[1]
        queued=score(path);hops=len(path)-1
        pipeline=size/bandwidth_B_ns+(hops-1)*min(size,packet_bytes)/bandwidth_B_ns+hops*latency_ns
        return dict(service_ns=pipeline+queued,queue_penalty_ns=queued,path=path,
            scope='packet pipeline plus observed committed-work ranking proxy; no future queue oracle')

    def cut_work(self,axis,index,forward=True):
        if axis not in ('row','col'):raise ValueError('unknown cut axis')
        limit=self.height if axis=='row' else self.width
        if not 0<index<limit:raise ValueError('cut outside mesh')
        selected=[]
        for row in self.edges:
            a,b=row[:2];u,v=(a//self.width,b//self.width) if axis=='row' else (a%self.width,b%self.width)
            if ((u<index<=v) if forward else (v<index<=u)):selected.append(row)
        return dict(announced_bytes=sum(r[2] for r in selected),queued_here_bytes=sum(r[3] for r in selected),
            scope='observed directed cut counters; no total-completion or global-infeasibility claim')
