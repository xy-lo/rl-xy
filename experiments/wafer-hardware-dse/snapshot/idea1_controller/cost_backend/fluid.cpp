// Same FIFO fluid GEMM event algorithm as the frozen Python reference.
// Host implementation acceleration only: same streams, comparisons and doubles.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <vector>
extern "C" {
struct Problem { int64_t k,ab,bb,ob,acc; };
struct Core { int64_t id,bc,mc,nc,tm,tn,tk; };
struct Hardware { double shared,load,store,gemm,vec; int64_t sram; };
struct Result { double ns,busy,work; int64_t active,reads,writes,flops,vectors,events,tails,workspace,peak; };
}
struct Action { char kind; int64_t a,b,c; };
struct Stream {
 Core d; Problem p; int64_t bi=0,mi=0,ni=0,ki=0; int phase=0;
 bool next(Action &a) {
  if(bi>=d.bc || d.mc<=0 || d.nc<=0)return false;
  int64_t mm=std::min(d.tm,d.mc-mi),nn=std::min(d.tn,d.nc-ni),kk=std::min(d.tk,p.k-ki);
  if(phase==0){a={'A',mm*kk*p.ab+kk*nn*p.bb+mm*nn*p.acc,int64_t(mm<d.tm||nn<d.tn||kk<d.tk),0};phase=1;}
  else if(phase==1){a={'R',mm*kk*p.ab+kk*nn*p.bb,0,0};phase=2;}
  else if(phase==2){a={'C',2*mm*nn*kk,0,0};ki+=d.tk;if(ki<p.k)phase=0;else phase=3;}
  else {a={'W',mm*nn*p.ob,0,0};phase=0;ki=0;ni+=d.tn;if(ni>=d.nc){ni=0;mi+=d.tm;if(mi>=d.mc){mi=0;bi++;}}}
  return true;
 }
};
struct Event { double when; int core; Action a; };
struct EventOrder { bool operator()(const Event&a,const Event&b) const {if(a.when!=b.when)return a.when>b.when;return a.core>b.core;} };
struct Flow { int core; double remaining,cap,arrival; int64_t serial; bool done=false; };
static bool flow_before(const Flow&a,const Flow&b){if(a.arrival!=b.arrival)return a.arrival<b.arrival;if(a.core!=b.core)return a.core<b.core;return a.serial<b.serial;}
extern "C" int idea1_fluid_gemm(const Problem *problem,const Core *core,int n,const Hardware *hw,Result*out,char*err) {
 try {
  *out=Result{};out->active=n;std::vector<Stream> streams;streams.reserve(n);
  int maxid=0;for(int i=0;i<n;i++){streams.push_back(Stream{core[i],*problem});maxid=std::max(maxid,int(core[i].id));}
  // Heap ties use actual core IDs. Store index lookup separately.
  std::vector<int> index(maxid+1,-1);for(int i=0;i<n;i++)index[core[i].id]=i;
  std::vector<double> completion(n,0.),work(n,0.);std::priority_queue<Event,std::vector<Event>,EventOrder> events;std::vector<Flow>flows;flows.reserve(n);
  auto next_action=[&](int cid,double when){Action a;int i=index[cid];if(streams[i].next(a))events.push(Event{when,cid,a});else completion[i]=when;};
  for(int i=0;i<n;i++)next_action(core[i].id,0.);
  double clock=0.,rounding_budget=0.;int64_t serial=0;std::vector<int> active;std::vector<double> rates;std::vector<int>done;
  active.reserve(n);rates.reserve(n);done.reserve(n);
  while(!events.empty()||!flows.empty()) {
   while(!events.empty()&&events.top().when<=clock+1e-8){auto e=events.top();events.pop();auto a=e.a;out->events++;
    if(a.kind=='A'){out->workspace=std::max(out->workspace,a.a);out->tails+=a.b;if(a.a>hw->sram){std::snprintf(err,512,"private workspace exceeds capacity");return 2;}next_action(e.core,e.when);}
    else if(a.kind=='C'){double duration=double(a.a)/hw->gemm+double(a.b)/hw->vec;out->flops+=a.a;out->vectors+=a.b;work[index[e.core]]+=duration;next_action(e.core,e.when+duration);}
    else if(a.kind=='R'||a.kind=='W'){
     Flow f{e.core,double(a.a),a.kind=='R'?hw->load:hw->store,e.when,++serial};
     if(flows.empty()||!flow_before(f,flows.back()))flows.push_back(f);
     else flows.insert(std::upper_bound(flows.begin(),flows.end(),f,[](const Flow&a,const Flow&b){return flow_before(a,b);}),f);
     if(a.kind=='R')out->reads+=a.a;else out->writes+=a.a;
    } else {std::snprintf(err,512,"unknown action");return 3;}
   }
   active.clear();rates.clear();double available=hw->shared;double flowdt=INFINITY;
   for(int i=0;i<int(flows.size());i++) {if(available<=0)break;double rate=std::min(flows[i].cap,available);active.push_back(i);rates.push_back(rate);available-=rate;flowdt=std::min(flowdt,flows[i].remaining/rate);}
   out->peak=std::max(out->peak,int64_t(active.size()));double event_time=events.empty()?INFINITY:events.top().when;
   double next_time=std::min(event_time,clock+flowdt);if(!std::isfinite(next_time)){std::snprintf(err,512,"no progress");return 4;}
   if(next_time<=clock && event_time>clock+1e-8 && flowdt>0.){
    next_time=std::nextafter(clock,INFINITY);rounding_budget+=next_time-clock;
    if(rounding_budget>1.){std::snprintf(err,512,"clock rounding budget exceeded 1 ns");return 6;}
   }
   double dt=std::max(0.,next_time-clock);if(!active.empty())out->busy+=dt;
   for(int j=0;j<int(active.size());j++)flows[active[j]].remaining-=dt*rates[j];clock=next_time;done.clear();
   for(int i:active)if(flows[i].remaining<=1e-5){flows[i].done=true;done.push_back(flows[i].core);}
   flows.erase(std::remove_if(flows.begin(),flows.end(),[](const Flow&f){return f.done;}),flows.end());
   for(int cid:done)next_action(cid,clock);
  }
  for(double t:completion)out->ns=std::max(out->ns,t);for(double t:work)out->work=std::max(out->work,t);return 0;
 }catch(...){std::snprintf(err,512,"C++ fluid reference exception");return 5;}
}
