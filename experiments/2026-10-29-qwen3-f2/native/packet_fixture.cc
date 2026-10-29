#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "astra-sim/workload/RlxyGpu.hh"
#include "astra-sim/workload/RlxyMesh.hh"
#include <astra-network-analytical/common/NetworkParser.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <cstdlib>
using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;
struct Completion {std::shared_ptr<EventQueue> queue;int count=0;uint64_t end=0;};
void done(void* p){auto* x=static_cast<Completion*>(p);if(++x->count!=1)std::abort();x->end=x->queue->get_current_time();}
struct Flow {int src,dst;uint64_t bytes;sim_request request;Completion send,recv;};
struct Fixture {std::shared_ptr<EventQueue> queue;std::vector<std::unique_ptr<CongestionAwareNetworkApi>> apis;std::vector<Flow> flows;};
void issue(void* p){
 auto& f=*static_cast<Fixture*>(p);
 for(size_t i=0;i<f.flows.size();++i){auto& x=f.flows[i];f.apis[x.dst]->sim_recv(nullptr,x.bytes,0,x.src,100+i,&x.request,&done,&x.recv);}
 for(size_t i=0;i<f.flows.size();++i){auto& x=f.flows[i];f.apis[x.src]->sim_send(nullptr,x.bytes,0,x.dst,100+i,&x.request,&done,&x.send);}
}
int main(int argc,char** argv){
 if(argc<4)return 2;
 Fixture f;f.queue=std::make_shared<EventQueue>();Topology::set_event_queue(f.queue);CommonNetworkApi::set_event_queue(f.queue);
 NetworkParser parser(argv[1]);std::shared_ptr<Topology> topology;
 if(std::string(argv[2])=="gpu")topology=std::make_shared<rlxy::GpuCluster>(parser);else topology=std::make_shared<rlxy::Mesh2D>(parser);
 CongestionAwareNetworkApi::set_topology(topology);
 for(int rank=0;rank<topology->get_npus_count();++rank)f.apis.emplace_back(std::make_unique<CongestionAwareNetworkApi>(rank));
 for(int i=3;i<argc;++i){std::istringstream in(argv[i]);int src,dst;uint64_t bytes;char c;in>>src>>c>>dst>>c>>bytes;Flow x{};x.src=src;x.dst=dst;x.bytes=bytes;x.send.queue=f.queue;x.recv.queue=f.queue;f.flows.push_back(x);}
 f.queue->schedule_event(1,&issue,&f);
 while(!f.queue->finished())f.queue->proceed();
 bool pass=true;std::cout<<"{\"flows\":[";
 for(size_t i=0;i<f.flows.size();++i){const auto& x=f.flows[i];if(i)std::cout<<",";pass&=x.send.count==1&&x.recv.count==1&&x.send.end==x.recv.end;
 std::cout<<"{\"src\":"<<x.src<<",\"dst\":"<<x.dst<<",\"bytes\":"<<x.bytes<<",\"send_callbacks\":"<<x.send.count<<",\"recv_callbacks\":"<<x.recv.count<<",\"end_tick\":"<<x.send.end<<"}";}
 std::cout<<"],\"status\":\""<<(pass?"PASS":"FAIL")<<"\"}\n";return pass?0:1;
}
