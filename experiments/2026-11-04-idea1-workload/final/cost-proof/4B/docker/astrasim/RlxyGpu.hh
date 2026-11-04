// MIT. GPU endpoint -> node NVSwitch -> shared node uplink -> spine.
// Native analytical Link/Chunk/EventQueue model transit and contention.
#pragma once
#include <astra-network-analytical/congestion_aware/Topology.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <stdexcept>
namespace rlxy {
class GpuCluster final : public NetworkAnalyticalCongestionAware::Topology {
    int per_node,nodes;
public:
    explicit GpuCluster(const NetworkAnalytical::NetworkParser& p) {
        const auto count=p.get_npus_counts_per_dim();
        const auto bw=p.get_bandwidths_per_dim();const auto lat=p.get_latencies_per_dim();
        if(count.size()!=2 || count[0]<2 || count[1]<2 || bw.size()!=2 || lat.size()!=2)
            throw std::runtime_error("GPU cluster needs GPUs/node and node count >=2");
        per_node=count[0];nodes=count[1];npus_count=per_node*nodes;
        devices_count=npus_count+nodes+1;dims_count=2;
        npus_count_per_dim=count;bandwidth_per_dim=bw;instantiate_devices();
        for(int gpu=0;gpu<npus_count;gpu++)connect(gpu,npus_count+gpu/per_node,bw[0],lat[0],true);
        for(int node=0;node<nodes;node++)connect(npus_count+node,npus_count+nodes,bw[1],lat[1],true);
    }
    NetworkAnalyticalCongestionAware::Route route(
        NetworkAnalytical::DeviceId src,NetworkAnalytical::DeviceId dst) const noexcept override {
        NetworkAnalyticalCongestionAware::Route result;result.push_back(devices[src]);
        if(src==dst)return result;
        const auto a=src/per_node,b=dst/per_node;
        result.push_back(devices[npus_count+a]);
        if(a!=b){result.push_back(devices[npus_count+nodes]);result.push_back(devices[npus_count+b]);}
        result.push_back(devices[dst]);return result;
    }
};
}
