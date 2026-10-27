
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-static-routing-helper.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <limits>

using namespace ns3;

struct Edge {
    uint32_t u, v;
    double capacityKbps;
    uint32_t interfaceU = 0, interfaceV = 0;
    Ipv4Address addressU, addressV;
};

static bool ReadForecast(
    const std::string& filename,
    std::vector<std::vector<double>>& forecast) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        try {
            while (std::getline(ss, cell, ','))
                row.push_back(std::stod(cell));
        } catch (...) {
            return false;
        }

        if (!row.empty()) forecast.push_back(row);
    }

    if (forecast.size() != 23) return false;
    for (const auto& row : forecast)
        if (row.size() != 23) return false;

    return true;
}

static std::vector<uint32_t> FindPath(
    uint32_t source, uint32_t destination,
    const std::vector<Edge>& edges,
    const std::vector<std::vector<double>>& forecast,
    bool proactive) {

    const uint32_t N = 23;
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double> dist(N, INF);
    std::vector<int> prev(N, -1);

    using Item = std::pair<double, uint32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

    dist[source] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();

        if (cost > dist[u]) continue;
        if (u == destination) break;

        for (const auto& e : edges) {
            uint32_t v;
            if (e.u == u) v = e.v;
            else if (e.v == u) v = e.u;
            else continue;

            double weight = 1.0;

            if (proactive) {
                // Forecast matrix values are used as a demand signal.
                // Capacity-aware penalty discourages routes with high
                // forecast demand relative to available link capacity.
                double capacity = std::max(1.0, e.capacityKbps);
                double demand = std::max(0.0, forecast[u][v]);
                double utilization = demand / capacity;

                // Bounded, non-negative penalty keeps Dijkstra valid.
                weight += 10.0 * utilization * utilization;
            }

            double candidate = cost + weight;
            if (candidate < dist[v]) {
                dist[v] = candidate;
                prev[v] = static_cast<int>(u);
                pq.push({candidate, v});
            }
        }
    }

    if (source != destination && prev[destination] == -1)
        return {};

    std::vector<uint32_t> path;
    for (int at = static_cast<int>(destination); at != -1;
         at = prev[at]) {
        path.push_back(static_cast<uint32_t>(at));
        if (static_cast<uint32_t>(at) == source) break;
    }

    std::reverse(path.begin(), path.end());
    if (path.empty() || path.front() != source) return {};
    return path;
}

int main(int argc, char* argv[]) {
    std::string edgeFile =
        "/workspaces/comp_net2/geant_edges.txt";
    std::string capacityFile =
        "/workspaces/comp_net2/geant_link_capacities.csv";
    std::string forecastFile =
        "/workspaces/comp_net2/ns3_forecast.csv";
    std::string mode = "proactive";
    std::string resultCsv =
        "/workspaces/comp_net2/ns3_results.csv";
    double duration = 10.0;
    double offeredRateKbps = 1000.0;

    const uint32_t source = 0;
    const uint32_t destination = 22;

    CommandLine cmd(__FILE__);
    cmd.AddValue("edges", "GÉANT edge file", edgeFile);
    cmd.AddValue("capacities", "Link capacity CSV", capacityFile);
    cmd.AddValue("forecast", "23x23 forecast CSV", forecastFile);
    cmd.AddValue("mode", "baseline or proactive", mode);
    cmd.AddValue("duration", "Simulation duration (seconds)", duration);
    cmd.AddValue("rate", "Identical offered rate for both modes (kbps)",
                 offeredRateKbps);
    cmd.AddValue("results", "Results CSV output path", resultCsv);
    cmd.Parse(argc, argv);

    if (mode != "baseline" && mode != "proactive") {
        std::cerr << "ERROR: mode must be baseline or proactive\n";
        return 1;
    }

    std::vector<std::vector<double>> forecast;
    if (!ReadForecast(forecastFile, forecast)) {
        std::cerr << "ERROR: Could not read a 23x23 forecast from "
                  << forecastFile << "\n";
        return 1;
    }

    std::ifstream edgeIn(edgeFile);
    if (!edgeIn) {
        std::cerr << "ERROR: Cannot open " << edgeFile << "\n";
        return 1;
    }

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    uint32_t u, v;
    while (edgeIn >> u >> v) {
        if (u < 23 && v < 23 && u != v)
            pairs.push_back({u, v});
    }

    if (pairs.empty()) {
        std::cerr << "ERROR: No valid topology edges\n";
        return 1;
    }

    // Capacity CSV schema:
    // u_idx,v_idx,u_source_id,v_source_id,capacity_kbps,igp_metric
    std::map<std::pair<uint32_t, uint32_t>, double> capacities;
    std::ifstream capIn(capacityFile);

    if (!capIn) {
        std::cerr << "ERROR: Cannot open " << capacityFile << "\n";
        return 1;
    }

    std::string line;
    std::getline(capIn, line); // header

    while (std::getline(capIn, line)) {
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;

        while (std::getline(ss, field, ','))
            fields.push_back(field);

        if (fields.size() < 5) continue;

        try {
            uint32_t a = std::stoul(fields[0]);
            uint32_t b = std::stoul(fields[1]);
            double cap = std::stod(fields[4]);

            if (a < 23 && b < 23 && cap > 0)
                capacities[{a, b}] = cap;
        } catch (...) {
            continue;
        }
    }

    std::vector<Edge> edges;
    for (const auto& p : pairs) {
        double cap = 1000000.0;
        auto it = capacities.find(p);

        if (it == capacities.end())
            it = capacities.find({p.second, p.first});

        if (it != capacities.end())
            cap = it->second;

        edges.push_back({p.first, p.second, cap});
    }

    auto path = FindPath(source, destination, edges,
                         forecast, mode == "proactive");

    if (path.size() < 2) {
        std::cerr << "ERROR: No route found from source to destination\n";
        return 1;
    }

    NodeContainer nodes;
    nodes.Create(23);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper address;
    uint32_t subnet = 1;

    for (auto& e : edges) {
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute(
            "DataRate",
            StringValue(std::to_string(e.capacityKbps) + "Kbps"));
        p2p.SetChannelAttribute("Delay", StringValue("5ms"));

        NetDeviceContainer dev =
            p2p.Install(nodes.Get(e.u), nodes.Get(e.v));

        std::ostringstream network;
        network << "10." << (subnet / 250) << "."
                << (subnet % 250 + 1) << ".0";

        address.SetBase(Ipv4Address(network.str().c_str()),
                        "255.255.255.0");
        Ipv4InterfaceContainer interfaces = address.Assign(dev);

        Ptr<Ipv4> ipv4U = nodes.Get(e.u)->GetObject<Ipv4>();
        Ptr<Ipv4> ipv4V = nodes.Get(e.v)->GetObject<Ipv4>();

        e.interfaceU = ipv4U->GetInterfaceForDevice(dev.Get(0));
        e.interfaceV = ipv4V->GetInterfaceForDevice(dev.Get(1));
        e.addressU = interfaces.GetAddress(0);
        e.addressV = interfaces.GetAddress(1);
        ++subnet;
    }

    Ipv4StaticRoutingHelper routingHelper;

    // Install routes along the selected forward path.
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        uint32_t current = path[i], next = path[i + 1];

        auto it = std::find_if(edges.begin(), edges.end(),
            [&](const Edge& e) {
                return (e.u == current && e.v == next) ||
                       (e.v == current && e.u == next);
            });

        if (it == edges.end()) {
            std::cerr << "ERROR: Missing edge on selected path\n";
            return 1;
        }

        Ipv4Address nextHop =
            it->u == current ? it->addressV : it->addressU;
        uint32_t iface =
            it->u == current ? it->interfaceU : it->interfaceV;

        Ptr<Ipv4StaticRouting> routing =
            routingHelper.GetStaticRouting(
                nodes.Get(current)->GetObject<Ipv4>());

        Ipv4Address destAddr =
            nodes.Get(destination)->GetObject<Ipv4>()
                ->GetAddress(1, 0).GetLocal();

        routing->AddHostRouteTo(destAddr, nextHop, iface);
    }

    // Install reverse routes.
    for (size_t i = path.size() - 1; i > 0; --i) {
        uint32_t current = path[i], next = path[i - 1];

        auto it = std::find_if(edges.begin(), edges.end(),
            [&](const Edge& e) {
                return (e.u == current && e.v == next) ||
                       (e.v == current && e.u == next);
            });

        if (it == edges.end()) continue;

        Ipv4Address nextHop =
            it->u == current ? it->addressV : it->addressU;
        uint32_t iface =
            it->u == current ? it->interfaceU : it->interfaceV;

        Ptr<Ipv4StaticRouting> routing =
            routingHelper.GetStaticRouting(
                nodes.Get(current)->GetObject<Ipv4>());

        Ipv4Address sourceAddr =
            nodes.Get(source)->GetObject<Ipv4>()
                ->GetAddress(1, 0).GetLocal();

        routing->AddHostRouteTo(sourceAddr, nextHop, iface);
    }

    Ipv4Address destinationAddress =
        nodes.Get(destination)->GetObject<Ipv4>()
            ->GetAddress(1, 0).GetLocal();

    const uint16_t port = 9000;

    PacketSinkHelper sink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sink.Install(nodes.Get(destination));
    sinkApp.Start(Seconds(0.5));
    sinkApp.Stop(Seconds(duration));

    OnOffHelper client("ns3::UdpSocketFactory",
        InetSocketAddress(destinationAddress, port));
    client.SetAttribute("DataRate",
        StringValue(std::to_string(offeredRateKbps) + "Kbps"));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    client.SetAttribute("OnTime",
        StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    client.SetAttribute("OffTime",
        StringValue("ns3::ConstantRandomVariable[Constant=0]"));

    ApplicationContainer clientApp = client.Install(nodes.Get(source));
    clientApp.Start(Seconds(1.0));
    clientApp.Stop(Seconds(duration - 0.1));

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    Simulator::Stop(Seconds(duration));
    Simulator::Run();
    monitor->CheckForLostPackets();

    const std::string xml =
        "/workspaces/comp_net2/geant-flowmon-" + mode + ".xml";
    monitor->SerializeToXmlFile(xml, true, true);

    uint64_t txBytes = 0, rxBytes = 0;
    uint64_t txPackets = 0, rxPackets = 0, lostPackets = 0;
    double delaySeconds = 0.0;

    for (const auto& item : monitor->GetFlowStats()) {
        const auto& s = item.second;
        txBytes += s.txBytes;
        rxBytes += s.rxBytes;
        txPackets += s.txPackets;
        rxPackets += s.rxPackets;
        lostPackets += s.lostPackets;
        delaySeconds += s.delaySum.GetSeconds();
    }

    std::cout << "\n=== GÉANT NS-3 Experiment ===\n"
              << "Mode: " << mode << "\n"
              << "Nodes: " << nodes.GetN() << "\n"
              << "Links: " << edges.size() << "\n"
              << "Offered rate (same for both modes): "
              << offeredRateKbps << " kbps\n"
              << "Selected path: ";

    for (size_t i = 0; i < path.size(); ++i) {
        if (i) std::cout << " -> ";
        std::cout << path[i];
    }

    std::cout << "\nForecast value for OD pair 0->22: "
              << forecast[source][destination] << "\n"
              << "Tx bytes: " << txBytes << "\n"
              << "Rx bytes: " << rxBytes << "\n"
              << "Tx packets: " << txPackets << "\n"
              << "Rx packets: " << rxPackets << "\n"
              << "Lost packets: " << lostPackets << "\n"
              << "Aggregate delay (seconds): " << delaySeconds << "\n"
              << "FlowMonitor XML: " << xml << "\n";

    // Append one comparable row per experiment.
    bool needsHeader = false;
    {
        std::ifstream check(resultCsv);
        needsHeader = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream csv(resultCsv, std::ios::app);
    if (!csv) {
        std::cerr << "WARNING: Could not write results CSV\n";
    } else {
        if (needsHeader) {
            csv << "mode,rate_kbps,tx_bytes,rx_bytes,tx_packets,"
                   "rx_packets,lost_packets,aggregate_delay_s,path\n";
        }

        csv << mode << "," << offeredRateKbps << ","
            << txBytes << "," << rxBytes << ","
            << txPackets << "," << rxPackets << ","
            << lostPackets << "," << delaySeconds << ",\"";

        for (size_t i = 0; i < path.size(); ++i) {
            if (i) csv << "->";
            csv << path[i];
        }
        csv << "\"\n";
    }

    Simulator::Destroy();
    return 0;
}
