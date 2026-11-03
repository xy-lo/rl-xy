/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/Chunk.h>
#include <cassert>
#include "astra-sim/workload/RlxyMesh.hh"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;

namespace {
// Physical reference granularity is 4 KiB. Other explicitly configured values
// are numerical packetization/coarsening experiments, never a bandwidth scale.
uint64_t packet_bytes() {
    static const auto value=[]() {
        const char* text=std::getenv("RLXY_NETWORK_PACKET_BYTES");
        const auto bytes=text?std::stoull(text):uint64_t{4096};
        if(!bytes || (bytes&(bytes-1)) || bytes>1048576)
            throw std::runtime_error("packet bytes must be a power of two <=1MiB");
        return bytes;
    }();
    return value;
}
struct PacketMessage;
struct PacketArrival {PacketMessage* message;uint64_t bytes;};
struct PacketMessage {
    std::shared_ptr<Topology> topology;
    std::shared_ptr<EventQueue> events;
    Route route;
    void* legacy_arg;
    uint64_t count,packet,sent=0,received=0,remaining,packets=0,start;
    static void arrived(void* pointer) {
        auto* arrival=static_cast<PacketArrival*>(pointer);
        auto* message=arrival->message;const auto bytes=arrival->bytes;delete arrival;
        if(!message->remaining || message->received+bytes>message->count)std::abort();
        message->received+=bytes;--message->remaining;
        if(message->remaining)return;
        if(message->sent!=message->count || message->received!=message->count)std::abort();
        if(const auto path=std::getenv("RLXY_NETWORK_PACKET_LOG")) {
            std::ofstream log(path,std::ios::app);
            log<<"{\"bytes\":"<<message->count<<",\"sent_bytes\":"<<message->sent
               <<",\"received_bytes\":"<<message->received<<",\"packets\":"<<message->packets
               <<",\"packet_bytes\":"<<message->packet<<",\"hops\":"<<message->route.size()-1
               <<",\"start_tick\":"<<message->start<<",\"end_tick\":"<<message->events->get_current_time()
               <<",\"aggregate_callbacks\":1}\n";
        }
        auto* legacy=message->legacy_arg;delete message;
        CommonNetworkApi::process_chunk_arrival(legacy);
    }
    static void inject(void* pointer) {
        auto* message=static_cast<PacketMessage*>(pointer);
        const auto bytes=std::min(message->packet,message->count-message->sent);
        message->sent+=bytes;++message->packets;
        auto* arrival=new PacketArrival{message,bytes};
        auto chunk=std::make_unique<Chunk>(bytes,message->route,&PacketMessage::arrived,arrival);
        message->topology->send(std::move(chunk));
        // Same-tick FIFO injection interleaves messages issued together rather
        // than placing one whole message ahead of every concurrent message.
        if(message->sent<message->count)
            message->events->schedule_event(message->events->get_current_time(),&PacketMessage::inject,message);
    }
};
}

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

CongestionAwareNetworkApi::CongestionAwareNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionAwareNetworkApi::sim_send(void* const buffer,
                                        const uint64_t count,
                                        const int type,
                                        const int dst,
                                        const int tag,
                                        sim_request* const request,
                                        void (*msg_handler)(void*),
                                        void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionAwareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // Segment the message before transmission. Every packet streams through
    // existing directed Link queues; the original whole-message tracker is
    // completed once, only after every payload byte reached the destination.
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());
    const auto route = topology->route(src, dst);
    if(auto* mesh=dynamic_cast<rlxy::Mesh2D*>(topology.get()))mesh->reserve_route(route,count);
    const auto packet=packet_bytes();
    if(!count || route.size()<2)throw std::runtime_error("invalid network message/route");
    auto* message=new PacketMessage{topology,event_queue,route,arg_ptr,count,packet,0,0,
        (count+packet-1)/packet,0,event_queue->get_current_time()};
    event_queue->schedule_event(event_queue->get_current_time(),&PacketMessage::inject,message);

    // return
    return 0;
}
