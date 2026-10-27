
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-static-routing-helper.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <limits>

using namespace ns3;

struct Edge {
    uint32_t u, v;
    double capacityKbps;
    uint32_t ifaceU = 0, ifaceV = 0;
    Ipv4Address addrU, addrV;
};

struct TrafficFlow {
    uint32_t src;
    uint32_t dst;
    uint16_t port;
};

bool ReadForecast(const std::string& filename,
                  std::vector<std::vector<double>>& forecast) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open forecast: " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        try {
            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stod(cell));
            }
        } catch (...) {
            std::cerr << "Invalid number in forecast CSV\n";
            return false;
        }

        if (!row.empty()) {
            forecast.push_back(row);
        }
    }

    if (forecast.size() != 23) {
        std::cerr << "Forecast must have 23 rows; found "
                  << forecast.size() << "\n";
        return false;
    }

    for (const auto& row : forecast) {
        if (row.size() != 23) {
            std::cerr << "Every forecast row must have 23 columns\n";
            return false;
        }
    }

    return true;
}

std::vector<uint32_t> FindPath(
    uint32_t source,
    uint32_t destination,
    const std::vector<Edge>& edges,
    const std::vector<std::vector<double>>& forecast,
    const std::vector<double>& forecastLoad,
    bool proactive) {

    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double> dist(23, INF);
    std::vector<int> previous(23, -1);
    std::vector<bool> visited(23, false);

    using Entry = std::pair<double, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>,
                        std::greater<Entry>> pq;

    dist[source] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty()) {
        auto current = pq.top();
        pq.pop();

        double currentCost = current.first;
        uint32_t u = current.second;

        if (visited[u]) {
            continue;
        }
        visited[u] = true;

        if (u == destination) {
            break;
        }

        for (size_t ei = 0; ei < edges.size(); ++ei) {
            const auto& edge = edges[ei];
            uint32_t v;

            if (edge.u == u) {
                v = edge.v;
            } else if (edge.v == u) {
                v = edge.u;
            } else {
                continue;
            }

            if (visited[v]) {
                continue;
            }

            double weight = 1.0;

            if (proactive) {
                double capacity = std::max(1.0, edge.capacityKbps);
                double demand = std::max(0.0, forecastLoad[ei]);
                double utilization = demand / capacity;
                weight += 100.0 * utilization * utilization;
            }

            double newCost = currentCost + weight;

            if (newCost < dist[v]) {
                dist[v] = newCost;
                previous[v] = static_cast<int>(u);
                pq.push({newCost, v});
            }
        }
    }

    if (source != destination && previous[destination] == -1) {
        return {};
    }

    std::vector<uint32_t> path;

    for (int node = static_cast<int>(destination);
         node != -1;
         node = previous[node]) {
        path.push_back(static_cast<uint32_t>(node));

        if (static_cast<uint32_t>(node) == source) {
            break;
        }
    }

    std::reverse(path.begin(), path.end());

    if (path.empty() || path.front() != source) {
        return {};
    }

    return path;
}

int main(int argc, char* argv[]) {
    std::string root = "/workspaces/comp_net2";

    std::string edgeFile = root + "/geant_edges.txt";
    std::string capacityFile =
        root + "/geant_link_capacities.csv";
    std::string forecastFile = root + "/ns3_forecast.csv";

    std::string mode = "baseline";

    // Keep the first test lightweight.
    double duration = 5.0;
    double rateKbps = 10000.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("mode", "baseline or proactive", mode);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("rate", "Rate per flow in kbps", rateKbps);
    cmd.AddValue("edges", "Topology edge file", edgeFile);
    cmd.AddValue("capacities", "Link capacity CSV", capacityFile);
    cmd.AddValue("forecast", "23 by 23 forecast CSV", forecastFile);
    cmd.Parse(argc, argv);

    if (mode != "baseline" && mode != "proactive") {
        std::cerr << "Mode must be baseline or proactive\n";
        return 1;
    }

    if (duration <= 1.1 || rateKbps <= 0) {
        std::cerr << "Duration must exceed 1.1 seconds and rate must be positive\n";
        return 1;
    }

    std::cout << "Loading inputs...\n";

    std::vector<std::vector<double>> forecast;
    if (!ReadForecast(forecastFile, forecast)) {
        return 1;
    }

    std::ifstream edgeInput(edgeFile);
    if (!edgeInput) {
        std::cerr << "Cannot open topology: " << edgeFile << "\n";
        return 1;
    }

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    uint32_t u, v;

    while (edgeInput >> u >> v) {
        if (u < 23 && v < 23 && u != v) {
            pairs.push_back({u, v});
        }
    }

    std::ifstream capacityInput(capacityFile);
    if (!capacityInput) {
        std::cerr << "Cannot open capacities: "
                  << capacityFile << "\n";
        return 1;
    }

    std::map<std::pair<uint32_t, uint32_t>, double> capacities;

    std::string line;
    std::getline(capacityInput, line); // Header

    while (std::getline(capacityInput, line)) {
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;

        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        if (fields.size() < 5) {
            continue;
        }

        try {
            uint32_t a = std::stoul(fields[0]);
            uint32_t b = std::stoul(fields[1]);
            double cap = std::stod(fields[4]);

            if (a < 23 && b < 23 && cap > 0) {
                capacities[{a, b}] = cap;
                capacities[{b, a}] = cap;
            }
        } catch (...) {
            // Ignore malformed CSV rows.
        }
    }

    std::vector<Edge> edges;

    for (const auto& pair : pairs) {
        double cap = 1000000.0;

        auto it = capacities.find(pair);
        if (it != capacities.end()) {
            cap = it->second;
        }

        edges.push_back({pair.first, pair.second, cap});
    }

    if (edges.empty()) {
        std::cerr << "No valid topology links were loaded\n";
        return 1;
    }

    std::cout << "Loaded " << edges.size() << " links\n";

    // Estimate physical-link load by routing every forecast OD demand
    // over a baseline shortest-hop path through the actual topology.
    // Forecast entries are OD demands, not direct physical-link loads.
    std::vector<double> forecastLoad(edges.size(), 0.0);
    for (uint32_t src = 0; src < 23; ++src) {
        for (uint32_t dst = 0; dst < 23; ++dst) {
            if (src == dst) continue;
            std::vector<int> prev(23, -1);
            std::queue<uint32_t> q;
            prev[src] = static_cast<int>(src);
            q.push(src);
            while (!q.empty() && prev[dst] == -1) {
                uint32_t a = q.front();
                q.pop();
                for (const auto& e : edges) {
                    uint32_t b;
                    if (e.u == a) b = e.v;
                    else if (e.v == a) b = e.u;
                    else continue;
                    if (prev[b] == -1) {
                        prev[b] = static_cast<int>(a);
                        q.push(b);
                    }
                }
            }
            if (prev[dst] == -1) continue;
            double demand = std::max(0.0, forecast[src][dst]);
            uint32_t node = dst;
            while (node != src) {
                uint32_t parent = static_cast<uint32_t>(prev[node]);
                for (size_t ei = 0; ei < edges.size(); ++ei) {
                    const auto& e = edges[ei];
                    if ((e.u == parent && e.v == node) ||
                        (e.v == parent && e.u == node)) {
                        forecastLoad[ei] += demand;
                        break;
                    }
                }
                node = parent;
            }
        }
    }

    std::cout << "Forecast-based physical-link loads estimated from OD paths.\n";

    // Four identical flow definitions are used in both modes.
    std::vector<TrafficFlow> flows = {
        {0, 22, 9000},
        {1, 21, 9001},
        {3, 19, 9002},
        {5, 16, 9003}
    };

    std::vector<std::vector<uint32_t>> paths;

    for (const auto& flow : flows) {
        auto path = FindPath(
            flow.src, flow.dst, edges, forecast, forecastLoad,
            mode == "proactive");

        if (path.size() < 2) {
            std::cerr << "No path for flow "
                      << flow.src << " -> " << flow.dst << "\n";
            return 1;
        }

        paths.push_back(path);
    }

    std::cout << "Selected routes:\n";

    for (size_t i = 0; i < flows.size(); ++i) {
        std::cout << flows[i].src << " -> " << flows[i].dst << ": ";

        for (size_t j = 0; j < paths[i].size(); ++j) {
            if (j != 0) {
                std::cout << " -> ";
            }
            std::cout << paths[i][j];
        }

        std::cout << "\n";
    }

    std::cout << "Creating network...\n";

    NodeContainer nodes;
    nodes.Create(23);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper address;
    uint32_t subnet = 1;

    for (auto& edge : edges) {
        PointToPointHelper p2p;

        p2p.SetDeviceAttribute(
            "DataRate",
            StringValue(std::to_string(edge.capacityKbps) + "Kbps"));

        p2p.SetChannelAttribute("Delay", StringValue("5ms"));

        NetDeviceContainer devices =
            p2p.Install(nodes.Get(edge.u), nodes.Get(edge.v));

        uint32_t third = subnet / 250;
        uint32_t fourth = subnet % 250 + 1;

        std::ostringstream network;
        network << "10." << third << "." << fourth << ".0";

        address.SetBase(
            Ipv4Address(network.str().c_str()),
            "255.255.255.0");

        Ipv4InterfaceContainer interfaces = address.Assign(devices);

        Ptr<Ipv4> ipv4U = nodes.Get(edge.u)->GetObject<Ipv4>();
        Ptr<Ipv4> ipv4V = nodes.Get(edge.v)->GetObject<Ipv4>();

        edge.ifaceU = ipv4U->GetInterfaceForDevice(devices.Get(0));
        edge.ifaceV = ipv4V->GetInterfaceForDevice(devices.Get(1));

        edge.addrU = interfaces.GetAddress(0);
        edge.addrV = interfaces.GetAddress(1);

        ++subnet;
    }

    std::cout << "Installing routes...\n";

    Ipv4StaticRoutingHelper routingHelper;

    for (size_t k = 0; k < flows.size(); ++k) {
        const auto& flow = flows[k];
        const auto& path = paths[k];

        Ptr<Ipv4> srcIpv4 =
            nodes.Get(flow.src)->GetObject<Ipv4>();
        Ptr<Ipv4> dstIpv4 =
            nodes.Get(flow.dst)->GetObject<Ipv4>();

        // Each node's first actual network interface is index 1.
        Ipv4Address sourceAddress =
            srcIpv4->GetAddress(1, 0).GetLocal();
        Ipv4Address destinationAddress =
            dstIpv4->GetAddress(1, 0).GetLocal();

        // Forward routes.
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            uint32_t current = path[i];
            uint32_t next = path[i + 1];

            auto it = std::find_if(
                edges.begin(), edges.end(),
                [&](const Edge& e) {
                    return (e.u == current && e.v == next) ||
                           (e.v == current && e.u == next);
                });

            if (it == edges.end()) {
                std::cerr << "Route link not found\n";
                return 1;
            }

            Ipv4Address nextHop =
                it->u == current ? it->addrV : it->addrU;

            uint32_t interfaceIndex =
                it->u == current ? it->ifaceU : it->ifaceV;

            routingHelper.GetStaticRouting(
                nodes.Get(current)->GetObject<Ipv4>())
                ->AddHostRouteTo(
                    destinationAddress, nextHop, interfaceIndex);
        }

        // Reverse routes for return traffic and network diagnostics.
        for (size_t i = path.size() - 1; i > 0; --i) {
            uint32_t current = path[i];
            uint32_t next = path[i - 1];

            auto it = std::find_if(
                edges.begin(), edges.end(),
                [&](const Edge& e) {
                    return (e.u == current && e.v == next) ||
                           (e.v == current && e.u == next);
                });

            if (it == edges.end()) {
                std::cerr << "Reverse route link not found\n";
                return 1;
            }

            Ipv4Address nextHop =
                it->u == current ? it->addrV : it->addrU;

            uint32_t interfaceIndex =
                it->u == current ? it->ifaceU : it->ifaceV;

            routingHelper.GetStaticRouting(
                nodes.Get(current)->GetObject<Ipv4>())
                ->AddHostRouteTo(
                    sourceAddress, nextHop, interfaceIndex);
        }
    }

    std::cout << "Installing applications...\n";

    for (const auto& flow : flows) {
        Ptr<Ipv4> dstIpv4 =
            nodes.Get(flow.dst)->GetObject<Ipv4>();

        Ipv4Address destinationAddress =
            dstIpv4->GetAddress(1, 0).GetLocal();

        PacketSinkHelper sink(
            "ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), flow.port));

        ApplicationContainer sinkApps =
            sink.Install(nodes.Get(flow.dst));

        sinkApps.Start(Seconds(0.1));
        sinkApps.Stop(Seconds(duration));

        OnOffHelper client(
            "ns3::UdpSocketFactory",
            InetSocketAddress(destinationAddress, flow.port));

        client.SetAttribute(
            "DataRate",
            StringValue(std::to_string(rateKbps) + "Kbps"));

        client.SetAttribute("PacketSize", UintegerValue(1024));

        client.SetAttribute(
            "OnTime",
            StringValue(
                "ns3::ConstantRandomVariable[Constant=1]"));

        client.SetAttribute(
            "OffTime",
            StringValue(
                "ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer clientApps =
            client.Install(nodes.Get(flow.src));

        clientApps.Start(Seconds(1.0));
        clientApps.Stop(Seconds(duration - 0.05));
    }

    FlowMonitorHelper monitorHelper;
    Ptr<FlowMonitor> monitor = monitorHelper.InstallAll();

    std::cout << "Starting simulation for "
              << duration << " seconds...\n";

    Simulator::Stop(Seconds(duration));
    Simulator::Run();

    std::cout << "Simulation completed. Collecting metrics...\n";

    monitor->CheckForLostPackets();

    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;
    uint64_t lostPackets = 0;
    double delaySumSeconds = 0.0;

    for (const auto& entry : monitor->GetFlowStats()) {
        const auto& stats = entry.second;

        txBytes += stats.txBytes;
        rxBytes += stats.rxBytes;
        txPackets += stats.txPackets;
        rxPackets += stats.rxPackets;
        lostPackets += stats.lostPackets;
        delaySumSeconds += stats.delaySum.GetSeconds();
    }

    std::string xmlFile =
        root + "/geant-multiflow-" + mode + ".xml";

    monitor->SerializeToXmlFile(xmlFile, true, true);

    std::cout << "\n========== GÉANT MULTI-FLOW RESULTS ==========\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Nodes: " << nodes.GetN() << "\n";
    std::cout << "Links: " << edges.size() << "\n";
    std::cout << "Flows: " << flows.size() << "\n";
    std::cout << "Rate per flow: " << rateKbps << " kbps\n";
    std::cout << "Simulation duration: " << duration << " seconds\n";
    std::cout << "Transmitted bytes: " << txBytes << "\n";
    std::cout << "Received bytes: " << rxBytes << "\n";
    std::cout << "Transmitted packets: " << txPackets << "\n";
    std::cout << "Received packets: " << rxPackets << "\n";
    std::cout << "Lost packets: " << lostPackets << "\n";
    std::cout << "Aggregate packet delay: "
              << delaySumSeconds << " seconds\n";
    std::cout << "FlowMonitor XML: " << xmlFile << "\n";
    std::cout << "==============================================\n";

    Simulator::Destroy();
    return 0;
}
