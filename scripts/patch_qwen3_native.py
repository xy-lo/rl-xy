"""Apply only the compact program extension after the existing rl-xy patches."""
from pathlib import Path
import argparse,shutil
p=argparse.ArgumentParser();p.add_argument('--astra-root',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];up=a.astra_root
shutil.copy2(root/'docker/astrasim/RlxySequence.hh',up/'astra-sim/workload/RlxySequence.hh')
f=up/'astra-sim/workload/Workload.cc';s=f.read_text();needle='typedef ChakraProtoMsg::NodeType ChakraNodeType;'
if '#include "astra-sim/workload/RlxySequence.hh"' not in s:s=s.replace(needle,'#include "astra-sim/workload/RlxySequence.hh"\n\n'+needle)
if 'rlxy::Sequence::start' not in s:s=s.replace('    if (sys->replay_only) {\n','    if (node->name().find("qwen3_sequence") != std::string::npos) {\n        rlxy::Sequence::start(this,node->id());return;\n    }\n    if (sys->replay_only) {\n',1)
f.write_text(s)
f=up/'astra-sim/common/Logging.cc';s=f.read_text();needle='    auto& logger_sinks = logger->sinks();'
if 'RLXY_SEQUENCE_PROGRAM' not in s:s=s.replace(needle,'    if (std::getenv("RLXY_SEQUENCE_PROGRAM") && logger_name != "workload") logger->set_level(spdlog::level::warn);\n'+needle)
f.write_text(s)
