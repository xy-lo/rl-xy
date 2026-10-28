// rl-xy extension, MIT: resource admission on AstraSim's event clock.
#pragma once
#include <json/json.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
namespace rlxy {
using json = nlohmann::json;
struct State {
    json nodes;
    json core_profiles;
    std::map<std::string, uint64_t> objects;
    std::set<std::string> ready;
    std::map<std::string,std::set<uint64_t>> readers;
    std::map<std::string,uint64_t> writers;
    std::map<uint64_t,uint64_t> scratch;
    std::vector<uint64_t> core_used;
    std::map<uint64_t,std::vector<uint64_t>> core_scratch;
    uint64_t private_capacity=0;
    uint64_t last_read_node=0;
    std::set<std::string> last_reads;
    bool read_ids_interned=false;
    std::vector<std::string> read_id_symbols;
    std::map<uint64_t,std::vector<std::string>> active_reads;
    uint64_t dram=0, capacity=0, sram=0, memory_owner=0;
    int64_t reserved=0;
    std::array<char,65536> log_buffer;
    std::ofstream log;
    bool eager_flush=[](){const auto* p=std::getenv("RLXY_RESOURCE_EAGER_FLUSH");return p && std::string(p)=="1";}();
    void record(const char* event,uint64_t node,uint64_t tick) {
        json row{{"event",event},{"node_id",node},{"tick",tick},{"dram_bytes",dram},
                  {"sram_bytes",sram},{"reserved_bytes",reserved},{"local_memory_owner",memory_owner}};
        if(!core_used.empty()) {
            size_t count=0;for(auto value:core_used)if(value)count++;
            if(count*4<=core_used.size()) {
                row["core_sram_nonzero"]=json::array();
                for(size_t core=0;core<core_used.size();core++)
                    if(core_used[core])row["core_sram_nonzero"].push_back(json::array({core,core_used[core]}));
            } else row["core_sram_bytes"]=core_used;
        }
        log<<row.dump()<<"\n";
        // Successful process shutdown closes every stream before validation.
        // Per-event flush affected only host I/O, never the simulated ticks.
        if(eager_flush)log.flush();
    }
    const json& contract(uint64_t node) { return nodes.at(std::to_string(node)); }
    const json& workspace(const json& n) {
        if(n.contains("core_workspace_bytes"))return n.at("core_workspace_bytes");
        return core_profiles.at(n.at("core_workspace_profile").get<std::string>());
    }
    void configure_read_id_symbols(const json& rank) {
        read_ids_interned=rank.contains("read_id_symbols");
        read_id_symbols.clear();
        if(!read_ids_interned)return;
        const auto& table=rank.at("read_id_symbols");
        if(!table.is_array())throw std::runtime_error("rlxy read ID symbols must be an array");
        std::set<std::string> seen;
        read_id_symbols.reserve(table.size());
        for(const auto& item:table) {
            if(!item.is_string())throw std::runtime_error("rlxy read ID symbol must be a string");
            const auto value=item.get<std::string>();
            if(value.empty() || !seen.insert(value).second)throw std::runtime_error("rlxy empty/duplicate read ID symbol");
            read_id_symbols.push_back(value);
        }
    }
    std::string decode_read_id(const json& id) const {
        if(!read_ids_interned)return id.get<std::string>();
        // Test the JSON numeric category before conversion: get<uint64_t>()
        // alone can coerce negative/signed/float values instead of rejecting.
        if(!id.is_number_unsigned())throw std::runtime_error("rlxy read ID index must be unsigned integer");
        const auto index=id.get<uint64_t>();
        if(index>=read_id_symbols.size())throw std::runtime_error("rlxy read ID index out of range");
        return read_id_symbols.at(static_cast<size_t>(index));
    }
    std::vector<std::string> decode_reads(const json& n) {
        if(n.contains("reads"))return n.at("reads").get<std::vector<std::string>>();
        const auto& delta=n.at("read_delta");
        if(delta.at("base_node").get<uint64_t>()!=last_read_node)throw std::runtime_error("rlxy read delta order mismatch");
        if(!delta.at("remove").is_array() || !delta.at("add").is_array())throw std::runtime_error("rlxy read delta IDs must be arrays");
        auto value=last_reads;
        for(const auto& id:delta.at("remove"))
            if(!value.erase(decode_read_id(id)))throw std::runtime_error("rlxy invalid read delta removal");
        for(const auto& id:delta.at("add"))
            if(!value.insert(decode_read_id(id)).second)throw std::runtime_error("rlxy duplicate read delta addition");
        return std::vector<std::string>(value.begin(),value.end());
    }
    bool available(uint64_t node) {
        const auto& n=contract(node);
        if(n.contains("read_delta") && n.at("read_delta").at("base_node").get<uint64_t>()!=last_read_node)return false;
        for(const auto& w:n.at("ready_outputs"))
            if(w.get<std::string>().find(".reference.")!=std::string::npos)
                throw std::runtime_error("rlxy immutable reference write");
        if(!core_used.empty()) {
            const auto& work=workspace(n);
            if(!work.is_array() || work.size()!=core_used.size())throw std::runtime_error("rlxy private SRAM core count mismatch");
            uint64_t total=0;
            for(size_t core=0;core<core_used.size();core++) {
                if(!work[core].is_number_unsigned())throw std::runtime_error("rlxy invalid private SRAM value");
                const auto bytes=work[core].get<uint64_t>();
                if(bytes>private_capacity)throw std::runtime_error("rlxy private SRAM capacity exceeded");
                total+=bytes;
                if(core_used[core]>private_capacity-bytes)return false;
            }
            if(total!=n.at("workspace_bytes").get<uint64_t>())throw std::runtime_error("rlxy private SRAM sum mismatch");
        } else {
            if(n.at("workspace_bytes").get<uint64_t>()>n.at("sram_capacity").get<uint64_t>())
                throw std::runtime_error("rlxy SRAM capacity exceeded");
            if(sram+n.at("workspace_bytes").get<uint64_t>()>n.at("sram_capacity").get<uint64_t>())return false;
        }
        if(n.at("local_memory").get<bool>() && memory_owner!=0)return false;
        for(const auto& id:decode_reads(n)) {
            if(!objects.count(id))throw std::runtime_error("rlxy read absent/freed tensor: "+id);
            if(!ready.count(id))return false;
        }
        for(const auto& w:n.at("ready_outputs")) {
            const auto id=w.get<std::string>();
            if(writers.count(id) || readers.count(id))return false;
        }
        int64_t future_reserved=reserved+n.value("reserve_before",int64_t(0));
        if(future_reserved<0)throw std::runtime_error("rlxy reservation consumed twice");
        if(uint64_t(future_reserved)>capacity || dram>capacity-uint64_t(future_reserved))
            throw std::runtime_error("rlxy KV reservation exceeds capacity");
        uint64_t demand=dram+uint64_t(future_reserved);
        for(auto it=n.at("allocations").begin();it!=n.at("allocations").end();++it) {
            if(objects.count(it.key()))throw std::runtime_error("rlxy duplicate allocation");
            const auto bytes=it.value().get<uint64_t>();
            if(bytes>capacity || demand>capacity-bytes)
                throw std::runtime_error("rlxy DRAM capacity exceeded before issue");
            demand+=bytes;
        }
        return true;
    }
    void start(uint64_t node,uint64_t tick) {
        if(!available(node))throw std::runtime_error("rlxy issue without admission");
        const auto& n=contract(node);
        reserved+=n.value("reserve_before",int64_t(0));
        for(auto it=n.at("allocations").begin();it!=n.at("allocations").end();++it) {
            objects[it.key()]=it.value().get<uint64_t>();dram+=objects[it.key()];
        }
        active_reads[node]=decode_reads(n);
        for(const auto& id:active_reads[node])readers[id].insert(node);
        last_reads=std::set<std::string>(active_reads[node].begin(),active_reads[node].end());last_read_node=node;
        for(const auto& w:n.at("ready_outputs")) {
            const auto id=w.get<std::string>();ready.erase(id);writers[id]=node;
        }
        scratch[node]=n.at("workspace_bytes").get<uint64_t>();sram+=scratch[node];
        if(!core_used.empty()) {
            core_scratch[node]=workspace(n).get<std::vector<uint64_t>>();
            for(size_t core=0;core<core_used.size();core++)core_used[core]+=core_scratch[node][core];
        } else if(sram>n.at("sram_capacity").get<uint64_t>())throw std::runtime_error("rlxy concurrent SRAM overflow");
        if(n.at("local_memory").get<bool>())memory_owner=node;
        record("issue",node,tick);
    }
    void finish(uint64_t node,uint64_t tick) {
        const auto& n=contract(node);
        if(!scratch.count(node))throw std::runtime_error("rlxy callback without issue");
        for(const auto& w:n.at("ready_outputs")) {
            const auto id=w.get<std::string>();
            if(!objects.count(id))throw std::runtime_error("rlxy READY without storage");
            if(!writers.count(id) || writers.at(id)!=node)throw std::runtime_error("rlxy incorrect writer");
            writers.erase(id);
            ready.insert(id);
        }
        for(const auto& id:active_reads.at(node)) {
            if(!readers.count(id) || !readers.at(id).erase(node))throw std::runtime_error("rlxy incorrect reader");
            if(readers.at(id).empty())readers.erase(id);
        }
        active_reads.erase(node);
        for(const auto& f:n.at("frees")) {
            const auto id=f.get<std::string>();
            if(!objects.count(id))throw std::runtime_error("rlxy double free");
            if(readers.count(id) || writers.count(id))throw std::runtime_error("rlxy free with active access");
            dram-=objects.at(id);objects.erase(id);ready.erase(id);
        }
        sram-=scratch.at(node);scratch.erase(node);
        if(!core_used.empty()) {
            for(size_t core=0;core<core_used.size();core++) {
                if(core_scratch.at(node)[core]>core_used[core])throw std::runtime_error("rlxy private SRAM underflow");
                core_used[core]-=core_scratch.at(node)[core];
            }
            core_scratch.erase(node);
        }
        reserved+=n.value("reserve_after",int64_t(0));
        if(reserved<0 || uint64_t(reserved)>capacity || dram>capacity-uint64_t(reserved))
            throw std::runtime_error("rlxy invalid reservation release");
        if(n.at("local_memory").get<bool>()) {
            if(memory_owner!=node)throw std::runtime_error("rlxy incorrect memory owner");
            memory_owner=0;
        }
        record("callback",node,tick);
    }
};
inline std::map<int,std::unique_ptr<State>>& states() {
    static std::map<int,std::unique_ptr<State>> value;return value;
}
inline void initialize_impl(int rank) {
    const char* path=std::getenv("RLXY_RESOURCE_CONTRACT");if(!path)return;
    std::ifstream input(path);if(!input)throw std::runtime_error("rlxy contract unavailable");
    json root;input>>root;
    if(root.at("schema_version")!="1.0")throw std::runtime_error("rlxy resource schema mismatch");
    json selected;
    if(root.contains("rank_files")) {
        std::ifstream rank_input(root.at("rank_files").at(std::to_string(rank)).get<std::string>());
        if(!rank_input)throw std::runtime_error("rlxy rank contract unavailable");
        rank_input>>selected;
    } else selected=root.at("ranks").at(std::to_string(rank));
    const auto& r=selected;
    auto s=std::make_unique<State>();s->capacity=r.at("dram_capacity").get<uint64_t>();s->nodes=r.at("nodes");
    s->configure_read_id_symbols(r);
    if(r.contains("cores_per_die")) {
        const auto cores=r.at("cores_per_die").get<uint64_t>();
        s->private_capacity=r.at("sram_bytes_per_core").get<uint64_t>();
        if(cores<2 || cores>65536 || !s->private_capacity)throw std::runtime_error("rlxy invalid private SRAM hardware");
        s->core_used.resize(cores,0);
        s->core_profiles=r.value("core_workspace_profiles",json::object());
    }
    for(auto it=r.at("initial_objects").begin();it!=r.at("initial_objects").end();++it) {
        auto bytes=it.value().get<uint64_t>();
        if(bytes>s->capacity || s->dram>s->capacity-bytes)throw std::runtime_error("rlxy initial DRAM overflow");
        s->objects[it.key()]=bytes;s->ready.insert(it.key());s->dram+=bytes;
    }
    s->log.rdbuf()->pubsetbuf(s->log_buffer.data(),s->log_buffer.size());
    s->log.open(root.at("log_prefix").get<std::string>()+"."+std::to_string(rank)+".jsonl");
    if(!s->log)throw std::runtime_error("rlxy resource log unavailable");
    s->record("initial",0,0);states()[rank]=std::move(s);
}
// Sys callback wrappers can swallow std::exception and return rc=0. Resource
// failures must instead terminate explicitly, preserving a deterministic rc.
template<class F> inline auto checked(F f)->decltype(f()) {
    try {return f();} catch(const std::exception& e) {
        std::cerr<<"[rlxy resource fatal] "<<e.what()<<std::endl;std::exit(2);
    }
}
inline void initialize(int rank) {checked([&](){initialize_impl(rank);});}
inline bool available(int rank,uint64_t node) {return checked([&](){return !states().count(rank)||states().at(rank)->available(node);});}
inline void start(int rank,uint64_t node,uint64_t tick) {checked([&](){if(states().count(rank))states().at(rank)->start(node,tick);});}
inline void finish(int rank,uint64_t node,uint64_t tick) {checked([&](){if(states().count(rank))states().at(rank)->finish(node,tick);});}
}
