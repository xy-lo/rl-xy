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
namespace rlxy {
using SJ=nlohmann::json;
class Sequence : public AstraSim::Callable {
    AstraSim::Workload* w;
    AstraSim::Sys* sys;
    uint64_t node;
    const SJ* segments;
    size_t segment=0,stage=0;
    uint64_t iteration=0,executed=0,comms=0,tag_counter=0;
    double compute=0,memory=0,network=0;
    uint64_t pending_start=0;
    bool pending_network=false;
    AstraSim::DataSet* dataset=nullptr;
    std::ofstream log;
    int group_id;
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
              {"compute_ns",compute},{"memory_ns",memory},{"communication_ns",network}};
        log<<x.dump()<<"\n";log.flush();
        auto data=new AstraSim::WorkloadLayerHandlerData;data->node_id=node;
        w->call(AstraSim::EventType::General,data);
        delete this;
    }
    void next() {
        while(segment<segments->size()) {
            const auto& sg=segments->at(segment);
            const auto& steps=sg.at("steps");
            if(stage==steps.size()) {
                stage=0;iteration++;
                if(iteration>=sg.at("repeat").get<uint64_t>()) {
                    SJ x={{"rank",sys->id},{"segment",segment},{"end_tick",AstraSim::Sys::boostedTick()},
                          {"compute_ns",compute},{"memory_ns",memory},{"communication_ns",network}};
                    log<<x.dump()<<"\n";log.flush();
                    segment++;iteration=0;
                }
                continue;
            }
            const auto& s=steps.at(stage++);executed++;
            const auto kind=s.at("kind").get<std::string>();
            pending_network=false;
            if(kind=="local") {
                if(s.contains("ranks")) {bool involved=false;for(auto r:s.at("ranks"))if(r.get<int>()==sys->id)involved=true;if(!involved)continue;}
                const double c=s.value("compute",0.0)+iteration*s.value("compute_delta",0.0);
                const double m=s.value("memory",0.0)+iteration*s.value("memory_delta",0.0);
                if(c<-.01||m<-.01)throw std::runtime_error("negative sequence local time");
                compute+=c;memory+=m;
                uint64_t ticks=static_cast<uint64_t>(std::ceil(std::max(0.0,c+m)))*1000;
                if(!ticks)continue;
                sys->register_event(this,AstraSim::EventType::General,nullptr,ticks);
                return;
            }
            const uint64_t bytes=s.at("bytes").get<uint64_t>();
            if(kind=="AR") {
                pending_network=true;pending_start=AstraSim::Sys::boostedTick();comms++;
                dataset=sys->generate_all_reduce(bytes,std::vector<bool>(1,true),w->comm_groups.at(group_id),0,node);
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
        if(dataset){auto old=dataset;dataset=nullptr;delete old;}
        next();
    }
};
}
