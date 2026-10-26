/*
 * ns3_wan_sim.cc - Abilene WAN with matrix-driven traffic and per-interval routing weights.
 *
 * STATUS: written against the NS-3 3.3x API but NOT compiled or run in the environment
 * where it was authored (NS-3 was not available there). Expect to fix small API issues.
 *
 * Inputs (made by export_ns3.py):
 *   demands.csv   : step,src,dst,mbps
 *   weights_*.csv : step,link,weight        (link index = order of ABILENE_LINKS in routing.py)
 *
 * Run (from the ns-3 root, after copying this file into scratch/):
 *   ./ns3 run "scratch/ns3_wan_sim --demands=ns3_input/demands.csv --weights=ns3_input/weights_static.csv    --out=static.csv"
 *   ./ns3 run "scratch/ns3_wan_sim --demands=ns3_input/demands.csv --weights=ns3_input/weights_reactive.csv  --out=reactive.csv"
 *   ./ns3 run "scratch/ns3_wan_sim --demands=ns3_input/demands.csv --weights=ns3_input/weights_proactive.csv --out=proactive.csv"
 *
 * Each step lasts --stepSec seconds. At the start of every step the controller's link weights are
 * applied as interface metrics and global routing is recomputed, so shortest-path routing follows
 * the controller's decision.
 */
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WanSim");

// Same node/link order as routing.py (ABILENE_NODES / ABILENE_LINKS).
static const int N_NODES = 12;
static const int LINKS[15][2] = {
    {1, 0},  {0, 4},  {0, 5},  {0, 11}, {2, 5},  {2, 8}, {3, 6}, {3, 9},
    {3, 10}, {4, 6},  {4, 7},  {5, 6},  {7, 9},  {8, 11}, {9, 10}};
static const int N_LINKS = 15;

struct LinkIf
{
    Ptr<Ipv4> ipA, ipB;
    uint32_t ifA, ifB;
};
static std::vector<LinkIf> g_links;
static std::map<int, std::vector<std::pair<int, int>>> g_weights; // step -> (link, weight)

static void
ApplyWeights(int step)
{
    for (auto& [link, w] : g_weights[step])
    {
        g_links[link].ipA->SetMetric(g_links[link].ifA, w);
        g_links[link].ipB->SetMetric(g_links[link].ifB, w);
    }
    Ipv4GlobalRoutingHelper::RecomputeRoutingTables();
}

int
main(int argc, char* argv[])
{
    std::string demandsFile = "ns3_input/demands.csv";
    std::string weightsFile = "ns3_input/weights_static.csv";
    std::string outFile = "result.csv";
    std::string linkRate = "100Mbps";
    std::string delay = "5ms";
    double stepSec = 10.0;
    uint32_t pktSize = 1000;

    CommandLine cmd(__FILE__);
    cmd.AddValue("demands", "demand CSV", demandsFile);
    cmd.AddValue("weights", "weights CSV", weightsFile);
    cmd.AddValue("out", "summary CSV", outFile);
    cmd.AddValue("linkRate", "point-to-point link rate", linkRate);
    cmd.AddValue("delay", "link propagation delay", delay);
    cmd.AddValue("stepSec", "seconds per traffic-matrix step", stepSec);
    cmd.Parse(argc, argv);

    Config::SetDefault("ns3::Ipv4GlobalRouting::RespondToInterfaceEvents", BooleanValue(true));

    NodeContainer nodes;
    nodes.Create(N_NODES);
    InternetStackHelper stack;
    stack.Install(nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(linkRate));
    p2p.SetChannelAttribute("Delay", StringValue(delay));
    p2p.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("100p"));

    Ipv4AddressHelper addr;
    std::vector<Ipv4InterfaceContainer> ifaces;
    for (int l = 0; l < N_LINKS; ++l)
    {
        NetDeviceContainer d = p2p.Install(nodes.Get(LINKS[l][0]), nodes.Get(LINKS[l][1]));
        std::ostringstream net;
        net << "10.1." << (l + 1) << ".0";
        addr.SetBase(net.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ic = addr.Assign(d);
        ifaces.push_back(ic);
        LinkIf li;
        li.ipA = nodes.Get(LINKS[l][0])->GetObject<Ipv4>();
        li.ipB = nodes.Get(LINKS[l][1])->GetObject<Ipv4>();
        li.ifA = ic.Get(0).second;
        li.ifB = ic.Get(1).second;
        g_links.push_back(li);
    }

    // Loopback-style address for each node = first interface address (used as flow destination).
    std::vector<Ipv4Address> nodeAddr(N_NODES);
    for (int n = 0; n < N_NODES; ++n)
    {
        nodeAddr[n] = nodes.Get(n)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ---- read controller weights and schedule them ----
    int maxStep = 0;
    {
        std::ifstream f(weightsFile);
        std::string line;
        std::getline(f, line); // header
        while (std::getline(f, line))
        {
            int s, l, w;
            char c;
            std::istringstream is(line);
            is >> s >> c >> l >> c >> w;
            g_weights[s].push_back({l, w});
            maxStep = std::max(maxStep, s);
        }
    }
    for (int s = 0; s <= maxStep; ++s)
    {
        Simulator::Schedule(Seconds(s * stepSec + 0.001), &ApplyWeights, s);
    }

    // ---- read demands and install one UDP flow per (step, src, dst) ----
    uint16_t port = 9000;
    ApplicationContainer sinks;
    {
        std::ifstream f(demandsFile);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line))
        {
            int s, src, dst;
            double mbps;
            char c;
            std::istringstream is(line);
            is >> s >> c >> src >> c >> dst >> c >> mbps;
            if (mbps <= 0)
            {
                continue;
            }
            PacketSinkHelper sink("ns3::UdpSocketFactory",
                                  InetSocketAddress(Ipv4Address::GetAny(), port));
            sinks.Add(sink.Install(nodes.Get(dst)));
            OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(nodeAddr[dst], port));
            onoff.SetConstantRate(DataRate(static_cast<uint64_t>(mbps * 1e6)), pktSize);
            onoff.SetAttribute("StartTime", TimeValue(Seconds(s * stepSec + 0.5)));
            onoff.SetAttribute("StopTime", TimeValue(Seconds((s + 1) * stepSec)));
            onoff.Install(nodes.Get(src));
            ++port;
        }
    }
    sinks.Start(Seconds(0));

    FlowMonitorHelper fmh;
    Ptr<FlowMonitor> monitor = fmh.InstallAll();

    Simulator::Stop(Seconds((maxStep + 1) * stepSec + 2));
    Simulator::Run();

    // ---- summary ----
    monitor->CheckForLostPackets();
    uint64_t tx = 0, rx = 0, lost = 0;
    double delaySum = 0;
    for (auto& [id, st] : monitor->GetFlowStats())
    {
        tx += st.txPackets;
        rx += st.rxPackets;
        lost += st.lostPackets;
        delaySum += st.delaySum.GetSeconds();
    }
    double loss = tx ? 100.0 * (tx - rx) / tx : 0;
    double avgDelayMs = rx ? 1000.0 * delaySum / rx : 0;
    double thrMbps = (rx * pktSize * 8.0) / (((maxStep + 1) * stepSec) * 1e6);

    std::ofstream o(outFile);
    o << "tx_packets,rx_packets,loss_percent,avg_delay_ms,throughput_mbps\n"
      << tx << "," << rx << "," << loss << "," << avgDelayMs << "," << thrMbps << "\n";
    std::cout << "tx=" << tx << " rx=" << rx << " loss=" << loss << "% avgDelay=" << avgDelayMs
              << "ms throughput=" << thrMbps << "Mbps\n";

    Simulator::Destroy();
    return 0;
}
