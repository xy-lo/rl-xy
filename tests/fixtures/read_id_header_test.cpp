// Test only: exercise the actual resource-header decoder without Astra clocks.
#include "docker/astrasim/RlxyResources.hh"
#include <algorithm>
int main() {
    try {
        rlxy::json rank; std::cin >> rank;
        rlxy::State state; state.configure_read_id_symbols(rank);
        std::vector<std::pair<uint64_t,std::string>> order;
        for (auto it=rank.at("nodes").begin(); it!=rank.at("nodes").end(); ++it)
            order.emplace_back(std::stoull(it.key()),it.key());
        std::sort(order.begin(),order.end());
        rlxy::json output=rlxy::json::object();
        for (const auto& entry:order) {
            const auto reads=state.decode_reads(rank.at("nodes").at(entry.second));
            state.last_reads=std::set<std::string>(reads.begin(),reads.end());
            state.last_read_node=entry.first; output[entry.second]=reads;
        }
        std::cout << output.dump() << std::endl; return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl; return 2;
    }
}
