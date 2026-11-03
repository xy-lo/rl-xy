// MIT. Message-boundary XY/YX routing with actual FIFO and committed traffic.
// This preserves the baseline unlimited-queue analytical network model.
#pragma once
#include <astra-network-analytical/congestion_aware/Topology.h>
#include <astra-network-analytical/congestion_aware/Link.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <json/json.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iterator>
#include <cmath>
namespace rlxy {
class Mesh2D final : public NetworkAnalyticalCongestionAware::Topology {
    int width,height; bool adaptive=false; double ticks_per_ns=1.0;
    mutable uint64_t calls=0,changes=0;
    mutable double predicted_wait_delta_ns=0;
    inline static Mesh2D* active=nullptr;
    using Route=NetworkAnalyticalCongestionAware::Route;
    Route path(int src,int dst,bool xy) const noexcept {
        Route r;auto current=src;r.push_back(devices[current]);
        auto x=[&](){while(current%width!=dst%width){current+=(current%width<dst%width?1:-1);r.push_back(devices[current]);}};
        auto y=[&](){while(current/width!=dst/width){current+=(current/width<dst/width?width:-width);r.push_back(devices[current]);}};
        if(xy){x();y();}else{y();x();}return r;
    }
    double work(const Route& r) const noexcept {
        double v=0;if(r.size()<2)return v;auto a=r.begin(),b=std::next(a);for(;b!=r.end();++a,++b)v+=(*a)->rlxy_link((*b)->get_id())->rlxy_committed_ticks();return v;
    }
public:
    explicit Mesh2D(const NetworkAnalytical::NetworkParser& p) {
        const auto counts=p.get_npus_counts_per_dim();const auto bw=p.get_bandwidths_per_dim();const auto lat=p.get_latencies_per_dim();
        if(counts.size()!=2||counts[0]<2||counts[1]<2)throw std::runtime_error("mesh needs two dimensions >=2");
        width=counts[0];height=counts[1];dims_count=2;npus_count=devices_count=width*height;
        npus_count_per_dim=counts;bandwidth_per_dim=bw;
        const char* policy=std::getenv("RLXY_MESH_POLICY");
        if(policy && std::string(policy)!="xy" && std::string(policy)!="adaptive_xy_yx")throw std::runtime_error("unsupported mesh policy");
        adaptive=policy && std::string(policy)=="adaptive_xy_yx";
        if(const char* text=std::getenv("RLXY_MESH_TICKS_PER_NS")){
            char* end=nullptr;ticks_per_ns=std::strtod(text,&end);
            if(!end||*end||!std::isfinite(ticks_per_ns)||ticks_per_ns<1.0)throw std::runtime_error("invalid native clock scale");
        }
        instantiate_devices();
        for(int y=0;y<height;y++)for(int x=0;x<width;x++){
            const int id=x+y*width;
            if(x+1<width)connect(id,id+1,bw[0],lat[0],true);
            if(y+1<height)connect(id,id+width,bw[1],lat[1],true);
        }
        active=this;
    }
    Route route(NetworkAnalytical::DeviceId src,NetworkAnalytical::DeviceId dst) const noexcept override {
        auto r=path(src,dst,true);++calls;
        if(adaptive && src%width!=dst%width && src/width!=dst/width){
            auto alternate=path(src,dst,false);const auto a=work(r),b=work(alternate);
            // A bounded heuristic, not an optimal-routing or completion-time claim.
            if(a-b>std::max(ticks_per_ns,0.05*a)){r=std::move(alternate);++changes;predicted_wait_delta_ns+=a-b;}
        }
        if(std::getenv("RLXY_MESH_ROUTE_LOG")){
            std::cout<<"[rlxy.mesh.route] src="<<src<<" dst="<<dst<<" path=";
            for(const auto& d:r)std::cout<<d->get_id()<<",";std::cout<<std::endl;
        }
        return r;
    }
    void reserve_route(const Route& r,uint64_t bytes) const noexcept {
        if(r.size()<2)return;auto a=r.begin(),b=std::next(a);for(;b!=r.end();++a,++b)(*a)->rlxy_link((*b)->get_id())->rlxy_reserve(bytes);
    }
    static nlohmann::json snapshot() {
        if(!active)return nullptr;
        nlohmann::json edges=nlohmann::json::array();
        for(int src=0;src<active->devices_count;src++)for(int delta:{-active->width,-1,1,active->width}){
            const int dst=src+delta;if(dst<0||dst>=active->devices_count)continue;
            const auto link=active->devices[src]->rlxy_link(dst);if(!link)continue;
            const auto announced=link->rlxy_announced_bytes(),queued=link->rlxy_queued_bytes();const auto wait=link->rlxy_queue_wait_ticks();
            if(announced||queued||wait)edges.push_back({src,dst,announced,queued,double(wait)/active->ticks_per_ns,link->rlxy_committed_ticks()/active->ticks_per_ns});
        }
        return {{"schema","rlxy.mesh-observed/2"},{"width",active->width},{"height",active->height},
          {"policy",active->adaptive?"adaptive_xy_yx":"xy"},{"route_calls",active->calls},{"alternate_routes",active->changes},
          {"native_ticks_per_ns",active->ticks_per_ns},{"predicted_wait_delta_ns",active->predicted_wait_delta_ns/active->ticks_per_ns},{"edges",edges},
          {"edge_fields",{"src","dst","announced_not_started_bytes","queued_here_bytes","actual_queue_wait_ns","committed_service_ns"}}};
    }
};
}
