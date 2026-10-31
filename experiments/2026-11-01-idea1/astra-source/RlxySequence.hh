// MIT. Compact repeated operator programs on the pinned ASTRA event/network APIs.
// Repeats save input graph space; every collective/P2P still executes natively.
#pragma once
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include <json/json.hpp>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
namespace rlxy {
using SJ=nlohmann::json;
class Sequence : public AstraSim::Callable {
    AstraSim::Workload* w;
    AstraSim::Sys* sys;
    uint64_t node;
    const SJ* segments;
    size_t segment=0,stage=0;
    uint64_t iteration=0,executed=0,comms=0,tag_counter=0;
    double compute=0,memory=0,network=0,gate_wait_ns=0;
    uint64_t gate_wait_start=0;
    bool pending_gate=false;
    uint64_t pending_start=0;
    bool pending_network=false;
    AstraSim::DataSet* dataset=nullptr;
    std::ofstream log;
    int group_id;
    SJ injected_steps=SJ::array();
    size_t injected_index=0;
    uint64_t injected_iteration=0,injected_repeat=1;
    SJ controller_rpc(const SJ& step) {
        const char* path=std::getenv("RLXY_CONTROLLER_SOCKET");
        if(!path || std::strlen(path)>=sizeof(sockaddr_un::sun_path))
            throw std::runtime_error("invalid controller socket path");
        int fd=::socket(AF_UNIX,SOCK_STREAM,0);
        if(fd<0)throw std::runtime_error("controller socket create failed");
        sockaddr_un addr{};addr.sun_family=AF_UNIX;std::strncpy(addr.sun_path,path,sizeof(addr.sun_path)-1);
        if(::connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0){::close(fd);throw std::runtime_error("controller connect failed");}
        SJ request={{"rank",sys->id},{"group_id",group_id},{"tick",AstraSim::Sys::boostedTick()}, {"segment",segment},{"iteration",iteration},{"event",step.at("event")}};
        std::string wire=request.dump()+"\n";
        size_t sent=0;
        while(sent<wire.size()) {auto n=::write(fd,wire.data()+sent,wire.size()-sent);if(n<=0){::close(fd);throw std::runtime_error("controller write failed");}sent+=n;}
        std::string reply;char buffer[16384];
        while(true){auto n=::read(fd,buffer,sizeof(buffer));if(n<=0){::close(fd);throw std::runtime_error("controller read failed");}reply.append(buffer,n);if(reply.size()>100000000){::close(fd);throw std::runtime_error("controller reply too large");}auto end=reply.find('\n');if(end!=std::string::npos){reply.resize(end);break;}}
        ::close(fd);return SJ::parse(reply);
    }

    struct GateWaiter { Sequence* sequence; uint64_t threshold; };
    struct GateState { uint64_t value=0; std::vector<GateWaiter> waiters; };
    static std::unordered_map<std::string,GateState>& gates() {
        static std::unordered_map<std::string,GateState> g; return g;
    }
    static void signal_gate(const std::string& key, uint64_t increment) {
        auto& gate=gates()[key];gate.value+=increment;
        std::vector<GateWaiter> remaining;
        for (const auto& waiter:gate.waiters) {
            if (gate.value >= waiter.threshold)
                waiter.sequence->sys->register_event(waiter.sequence,AstraSim::EventType::General,nullptr,1);
            else remaining.push_back(waiter);
        }
        gate.waiters.swap(remaining);
    }

    static SJ& source() {
        static SJ j=[](){
            auto p=std::getenv("RLXY_SEQUENCE_PROGRAM");
            if(!p)throw std::runtime_error("sequence program not configured");
            std::ifstream f(p);if(!f)throw std::runtime_error("cannot open sequence program");
            SJ x;f>>x;return x;
        }();
        return j;
    }
    static void packet_done(void* p) { static_cast<Sequence*>(p)->call(AstraSim::EventType::General,nullptr); }
    void finish() {
        SJ x={{"rank",sys->id},{"status","PASS"},{"end_tick",AstraSim::Sys::boostedTick()},
              {"executed_steps",executed},{"communication_steps",comms},
              {"compute_ns",compute},{"memory_ns",memory},{"communication_ns",network},{"dependency_wait_ns",gate_wait_ns}};
        log<<x.dump()<<"\n";log.flush();
        auto data=new AstraSim::WorkloadLayerHandlerData;data->node_id=node;
        w->call(AstraSim::EventType::General,data);
        delete this;
    }
    void next() {
        while(segment<segments->size()) {
            const auto& sg=segments->at(segment);
            const auto& steps=sg.at("steps");
            if(!injected_steps.empty() && injected_index>=injected_steps.size()) {
                if(injected_iteration+1<injected_repeat){injected_iteration++;injected_index=0;}
                else {injected_steps=SJ::array();injected_index=0;injected_iteration=0;injected_repeat=1;}
            }
            if(stage==steps.size() && injected_index>=injected_steps.size()) {
                stage=0;iteration++;
                if(iteration>=sg.at("repeat").get<uint64_t>()) {
                    SJ x={{"rank",sys->id},{"segment",segment},{"end_tick",AstraSim::Sys::boostedTick()},
                          {"compute_ns",compute},{"memory_ns",memory},{"communication_ns",network},{"dependency_wait_ns",gate_wait_ns}};
                    log<<x.dump()<<"\n";log.flush();
                    segment++;iteration=0;
                }
                continue;
            }
            const SJ* step_ptr=nullptr;
            bool from_injected=injected_index<injected_steps.size();
            if(from_injected) step_ptr=&injected_steps.at(injected_index++);
            else {injected_steps=SJ::array();injected_index=0;step_ptr=&steps.at(stage++);}
            const auto& s=*step_ptr;executed++;
            const auto kind=s.at("kind").get<std::string>();
            if(kind=="controller_rpc") {
                if(injected_index<injected_steps.size())throw std::runtime_error("nested controller injection unsupported");
                auto reply=controller_rpc(s);
                if(!reply.contains("steps") || !reply.at("steps").is_array())throw std::runtime_error("controller missing steps");
                if(reply.value("finish",false)) {if(!reply.at("steps").empty())throw std::runtime_error("finish with injected steps");iteration=sg.at("repeat").get<uint64_t>()-1;stage=steps.size();}
                injected_steps=reply.at("steps");injected_index=0;injected_iteration=0;injected_repeat=reply.value("repeat",uint64_t(1));
                if(injected_repeat==0)throw std::runtime_error("invalid injected repeat");
                continue;
            }
            pending_network=false;
            if(kind=="local") {
                if(s.contains("ranks")) {bool involved=false;for(auto r:s.at("ranks"))if(r.get<int>()==sys->id)involved=true;if(!involved)continue;}
                const auto iter=from_injected ? injected_iteration : iteration;
                const double c=s.value("compute",0.0)+iter*s.value("compute_delta",0.0);
                const double m=s.value("memory",0.0)+iter*s.value("memory_delta",0.0);
                if(c<-.01||m<-.01)throw std::runtime_error("negative sequence local time");
                compute+=c;memory+=m;
                uint64_t ticks=static_cast<uint64_t>(std::ceil(std::max(0.0,c+m)))*1000;
                if(!ticks)continue;
                sys->register_event(this,AstraSim::EventType::General,nullptr,ticks);
                return;
            }
            if(kind=="gate_signal") {
                if(s.contains("rank") && s.at("rank").get<int>()!=sys->id) continue;
                if(std::getenv("RLXY_DEBUG_GATES"))std::fprintf(stderr,"signal rank=%d key=%s tick=%llu\n",sys->id,s.at("key").get<std::string>().c_str(),(unsigned long long)AstraSim::Sys::boostedTick());
                signal_gate(s.at("key").get<std::string>(),s.value("increment",uint64_t(1)));
                continue;
            }
            if(kind=="gate_wait") {
                const auto key=s.at("key").get<std::string>();
                const auto threshold=s.at("threshold").get<uint64_t>();
                auto& gate=gates()[key];
                if(std::getenv("RLXY_DEBUG_GATES"))std::fprintf(stderr,"wait rank=%d key=%s value=%llu threshold=%llu tick=%llu\n",sys->id,key.c_str(),(unsigned long long)gate.value,(unsigned long long)threshold,(unsigned long long)AstraSim::Sys::boostedTick());
                if(gate.value >= threshold) continue;
                gate_wait_start=AstraSim::Sys::boostedTick();pending_gate=true;
                gate.waiters.push_back({this,threshold});
                return;
            }
            const uint64_t bytes=s.at("bytes").get<uint64_t>();
            if(kind=="p2p_direct") {
                int src=s.at("src").get<int>(),dst=s.at("dst").get<int>();
                const int tag=s.at("tag").get<int>();
                if(src==dst || tag<0 || tag>=500000000) throw std::runtime_error("invalid direct p2p");
                if(sys->id!=src && sys->id!=dst)continue;
                pending_network=true;pending_start=AstraSim::Sys::boostedTick();comms++;
                AstraSim::sim_request req;req.srcRank=src;req.dstRank=dst;req.reqType=AstraSim::UINT8;
                if(sys->id==src)sys->front_end_sim_send(0,AstraSim::Sys::dummy_data,bytes,AstraSim::UINT8,dst,tag,&req,AstraSim::Sys::FrontEndSendRecvType::NATIVE,&Sequence::packet_done,this);
                else sys->front_end_sim_recv(0,AstraSim::Sys::dummy_data,bytes,AstraSim::UINT8,src,tag,&req,AstraSim::Sys::FrontEndSendRecvType::NATIVE,&Sequence::packet_done,this);
                return;
            }
            if(kind=="AR" || kind=="AG" || kind=="RS") {
                if(s.contains("ranks")) {bool involved=false;for(auto r:s.at("ranks"))if(r.get<int>()==sys->id)involved=true;if(!involved)continue;}
                pending_network=true;pending_start=AstraSim::Sys::boostedTick();comms++;
                const int group=s.value("group",group_id);
                if(kind=="AR") dataset=sys->generate_all_reduce(bytes,std::vector<bool>(1,true),w->comm_groups.at(group),0,node);
                else if(kind=="AG") dataset=sys->generate_all_gather(bytes,std::vector<bool>(1,true),w->comm_groups.at(group),0,node);
                else dataset=sys->generate_reduce_scatter(bytes,std::vector<bool>(1,true),w->comm_groups.at(group),0,node);
                dataset->set_notifier(this,AstraSim::EventType::CollectiveCommunicationFinished);
                return;
            }
            if(kind=="p2p") {
                int src=s.at("src").get<int>(),dst=s.at("dst").get<int>();
                tag_counter++;
                if(sys->id!=src&&sys->id!=dst)continue;
                pending_network=true;pending_start=AstraSim::Sys::boostedTick();comms++;
                AstraSim::sim_request req;req.srcRank=src;req.dstRank=dst;req.reqType=AstraSim::UINT8;
                const int tag=(group_id*10000000+tag_counter)%500000000;
                if(sys->id==src)sys->front_end_sim_send(0,AstraSim::Sys::dummy_data,bytes,AstraSim::UINT8,dst,tag,&req,AstraSim::Sys::FrontEndSendRecvType::NATIVE,&Sequence::packet_done,this);
                else sys->front_end_sim_recv(0,AstraSim::Sys::dummy_data,bytes,AstraSim::UINT8,src,tag,&req,AstraSim::Sys::FrontEndSendRecvType::NATIVE,&Sequence::packet_done,this);
                return;
            }
            throw std::runtime_error("unknown sequence step");
        }
        finish();
    }
public:
    Sequence(AstraSim::Workload* wl,uint64_t id):w(wl),sys(wl->sys),node(id) {
        auto& j=source();auto key=j.at("ranks").at(std::to_string(sys->id)).get<std::string>();
        const auto& pg=j.at("programs").at(key);group_id=pg.at("group").get<int>();segments=&pg.at("segments");
        const auto prefix=std::getenv("RLXY_SEQUENCE_LOG_PREFIX");
        if(!prefix)throw std::runtime_error("sequence log prefix missing");
        log.open(std::string(prefix)+"."+std::to_string(sys->id)+".jsonl");
        if(!log)throw std::runtime_error("cannot open sequence log");
    }
    static void start(AstraSim::Workload* w,uint64_t node) { (new Sequence(w,node))->next(); }
    void call(AstraSim::EventType event,AstraSim::CallData* data) override {
        if(pending_network)network+=(AstraSim::Sys::boostedTick()-pending_start)/1000.0;
        if(pending_gate){gate_wait_ns+=(AstraSim::Sys::boostedTick()-gate_wait_start)/1000.0;pending_gate=false;}
        if(dataset){auto old=dataset;dataset=nullptr;delete old;}
        next();
    }
};
}
