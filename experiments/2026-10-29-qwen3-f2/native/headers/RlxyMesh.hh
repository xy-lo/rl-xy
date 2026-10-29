// rl-xy extension, MIT. Rectangular 2D mesh with deterministic X-then-Y routing.
// Native Astra analytical Link/Chunk/EventQueue own all transit and contention.
#pragma once
#include <astra-network-analytical/congestion_aware/Topology.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
namespace rlxy {
class Mesh2D final : public NetworkAnalyticalCongestionAware::Topology {
    int width,height;
public:
    explicit Mesh2D(const NetworkAnalytical::NetworkParser& p) {
        const auto counts=p.get_npus_counts_per_dim();
        const auto bw=p.get_bandwidths_per_dim();
        const auto lat=p.get_latencies_per_dim();
        if(counts.size()!=2 || counts[0]<2 || counts[1]<2)
            throw std::runtime_error("rlxy mesh requires two dimensions >=2");
        width=counts[0];height=counts[1];dims_count=2;
        npus_count=devices_count=width*height;
        npus_count_per_dim=counts;bandwidth_per_dim=bw;
        instantiate_devices();
        for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
            const auto id=x+y*width;
            if(x+1<width)connect(id,id+1,bw[0],lat[0],true);
            if(y+1<height)connect(id,id+width,bw[1],lat[1],true);
        }
    }
    NetworkAnalyticalCongestionAware::Route route(
        NetworkAnalytical::DeviceId src,NetworkAnalytical::DeviceId dst) const noexcept override {
        NetworkAnalyticalCongestionAware::Route result;
        result.push_back(devices[src]);
        auto current=src;
        while(current%width!=dst%width) {
            current+=(current%width<dst%width?1:-1);result.push_back(devices[current]);
        }
        while(current/width!=dst/width) {
            current+=(current/width<dst/width?width:-width);result.push_back(devices[current]);
        }
        // Packet route traces are opt-in diagnostics, never the long-run default.
        if(std::getenv("RLXY_MESH_ROUTE_LOG")) {
            std::cout<<"[rlxy.mesh.route] src="<<src<<" dst="<<dst<<" path=";
            for(const auto& device:result)std::cout<<device->get_id()<<",";
            std::cout<<std::endl;
        }
        return result;
    }
};
}
