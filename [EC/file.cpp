#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>    
#include <chrono>    

using namespace std;

// --- Global Data Structures ---

struct Node {
    int id;
    int x, y;
    int cost;
};

struct ResultStats {
    long long min_Z = numeric_limits<long long>::max();
    long long max_Z = numeric_limits<long long>::min();
    long long sum_Z = 0;
    int count = 0;
    vector<int> best_path;
};

int N = 0;
int K = 0;
vector<Node> all_nodes;
vector<vector<int>> DistanceMatrix;

// --- Helper Functions ---

// Calculates Euclidean distance rounded to the nearest integer
int calculateDistance(int x1, int y1, int x2, int y2) {
    double dist = sqrt(pow((double)x1 - x2, 2) + pow((double)y1 - y2, 2));
    return static_cast<int>(round(dist));
}

// Initializes the N x N distance matrix
void initializeDistanceMatrix() {
    DistanceMatrix.assign(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            DistanceMatrix[i][j] = calculateDistance(
                all_nodes[i].x, all_nodes[i].y,
                all_nodes[j].x, all_nodes[j].y
            );
        }
    }
}

// Calculates the final objective function value (Z)
long long calculateObjectiveValue(const vector<int>& path) {
    if ((int)path.size() != K) return -1;

    long long total_cost = 0;
    long long total_distance = 0;

    // Total Cost (Sum of selected node costs)
    for (int node_id : path) {
        total_cost += all_nodes[node_id].cost;
    }

    // Total Edge Length (Cycle distance, closing the path)
    for (size_t i = 0; i < path.size(); i++) {
        int from_node = path[i];
        int to_node = path[(i + 1) % path.size()];
        total_distance += DistanceMatrix[from_node][to_node];
    }

    return total_cost + total_distance;
}

// Loads node data from a semicolon-separated CSV file
void loadInstanceData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file " << filename << endl;
        exit(1);
    }

    all_nodes.clear();
    string line;
    int id_counter = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        string segment;
        vector<int> values;

        // Read segments separated by semicolon (;)
        while (getline(ss, segment, ';')) {
            try {
                values.push_back(stoi(segment));
            }
            catch (const exception& e) {
                // Ignore parsing errors
            }
        }

        if (values.size() >= 3) {
            all_nodes.push_back({ id_counter++, values[0], values[1], values[2] });
        }
    }

    N = all_nodes.size();
    K = (N + 1) / 2; // Selection rule: N/2 rounded up

    if (N == 0) {
        cerr << "ERROR: No node data loaded from file." << endl;
        exit(1);
    }

    initializeDistanceMatrix();
    cout << "Instance " << filename << " loaded. Total nodes (N)=" << N << ", Selected nodes (K)=" << K << endl;
}

// Updates statistics after each run
void updateStats(ResultStats& stats, const vector<int>& path, long long Z) {
    stats.count++;
    stats.sum_Z += Z;

    if (Z < stats.min_Z) {
        stats.min_Z = Z;
        stats.best_path = path;
    }
    if (Z > stats.max_Z) {
        stats.max_Z = Z;
    }
}

// --- VISUALIZATION EXPORT FUNCTION ---
void exportVisualizationData(const string& instance_name, const string& method_name, const vector<int>& best_path, long long min_Z) {
    // Filename example: VIS_TSPA_Greedy_Cycle.txt
    string filename = "VIS_" + instance_name + "_" + method_name + ".txt";
    ofstream outfile(filename);

    if (!outfile.is_open()) {
        cerr << "ERROR: Could not create export file " << filename << endl;
        return;
    }

    // Header for plotting tools (ID;X;Y;Cost;Path_Order;Is_Selected;Min_Z)
    outfile << "ID;X;Y;Cost;Path_Order;Is_Selected;Min_Z\n";

    // Create a map of Path Order (which node is at which position 0-99)
    vector<int> path_order_map(N, -1);
    for (int i = 0; i < (int)best_path.size(); i++) {
        path_order_map[best_path[i]] = i;
    }

    // Export data for ALL N nodes
    for (int i = 0; i < N; i++) {
        bool is_selected = (path_order_map[i] != -1);

        outfile << i << ";"
            << all_nodes[i].x << ";"
            << all_nodes[i].y << ";"
            << all_nodes[i].cost << ";"
            << path_order_map[i] << ";"
            << (is_selected ? "1" : "0") << ";"
            << min_Z << "\n";
    }

    outfile.close();
    cout << "Data exported for visualization: " << filename << "\n";
}


// --- Implementation of Heuristics ---

// 1. Random Solution
pair<vector<int>, long long> randomSolution() {
    vector<int> all_indices(N);
    iota(all_indices.begin(), all_indices.end(), 0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);

    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    vector<int> path(all_indices.begin(), all_indices.begin() + K);

    // Randomly determine the path order
    std::shuffle(path.begin(), path.end(), rng);

    long long Z = calculateObjectiveValue(path);
    return { path, Z };
}

// 2. Nearest Neighbor (End)
pair<vector<int>, long long> nearestNeighborEnd(int startNode) {
    vector<int> path = { startNode };
    vector<bool> is_selected(N, false);
    is_selected[startNode] = true;

    while ((int)path.size() < K) {
        int last_node = path.back();
        int best_candidate = -1;
        long long min_delta_Z = numeric_limits<long long>::max();

        // Find candidate that minimizes Delta Z (Cost + New Edge)
        for (int j = 0; j < N; j++) {
            if (!is_selected[j]) {
                long long delta_Z = all_nodes[j].cost + DistanceMatrix[last_node][j];

                if (delta_Z < min_delta_Z) {
                    min_delta_Z = delta_Z;
                    best_candidate = j;
                }
            }
        }

        if (best_candidate != -1) {
            path.push_back(best_candidate);
            is_selected[best_candidate] = true;
        }
        else {
            break;
        }
    }

    long long Z = calculateObjectiveValue(path);
    return { path, Z };
}

// 3. Nearest Neighbor (All Positions)
pair<vector<int>, long long> nearestNeighborAll(int startNode) {
    vector<int> path = { startNode };
    vector<bool> is_selected(N, false);
    is_selected[startNode] = true;

    while ((int)path.size() < K) {
        int best_j = -1;
        int best_k = -1;
        long long min_delta_Z = numeric_limits<long long>::max();

        // Find candidate (j) and position (k) that minimizes Delta Z
        for (int j = 0; j < N; j++) {
            if (!is_selected[j]) {
                for (size_t k = 0; k < path.size(); k++) {
                    int p_prev = path[k];
                    int p_next = path[(k + 1) % path.size()];

                    // Delta Z: Cost_j + (D_prev,j + D_j,next) - D_prev,next
                    long long delta_Z = all_nodes[j].cost;
                    delta_Z += DistanceMatrix[p_prev][j];
                    delta_Z += DistanceMatrix[j][p_next];
                    delta_Z -= DistanceMatrix[p_prev][p_next];

                    if (delta_Z < min_delta_Z) {
                        min_delta_Z = delta_Z;
                        best_j = j;
                        best_k = k + 1;
                    }
                }
            }
        }

        if (best_j != -1) {
            path.insert(path.begin() + best_k, best_j);
            is_selected[best_j] = true;
        }
        else {
            break;
        }
    }

    long long Z = calculateObjectiveValue(path);
    return { path, Z };
}

// 4. Greedy Cycle
pair<vector<int>, long long> greedyCycle(int startNode) {
    vector<bool> is_selected(N, false);

    // 1. Initialization: Find the best second node for a minimal 2-node cycle
    int second_node = -1;
    long long min_start_Z = numeric_limits<long long>::max();

    for (int j = 0; j < N; j++) {
        if (j != startNode) {
            long long current_Z = all_nodes[startNode].cost + all_nodes[j].cost +
                2 * DistanceMatrix[startNode][j];

            if (current_Z < min_start_Z) {
                min_start_Z = current_Z;
                second_node = j;
            }
        }
    }

    if (second_node == -1) {
        return { {}, -1 };
    }

    vector<int> path = { startNode, second_node };
    is_selected[startNode] = is_selected[second_node] = true;

    // 2. Expansion using the same insertion logic as NN (All Positions)
    while ((int)path.size() < K) {
        int best_j = -1;
        int best_k = -1;
        long long min_delta_Z = numeric_limits<long long>::max();

        for (int j = 0; j < N; j++) {
            if (!is_selected[j]) {
                for (size_t k = 0; k < path.size(); k++) {
                    int p_prev = path[k];
                    int p_next = path[(k + 1) % path.size()];

                    long long delta_Z = all_nodes[j].cost;
                    delta_Z += DistanceMatrix[p_prev][j];
                    delta_Z += DistanceMatrix[j][p_next];
                    delta_Z -= DistanceMatrix[p_prev][p_next];

                    if (delta_Z < min_delta_Z) {
                        min_delta_Z = delta_Z;
                        best_j = j;
                        best_k = k + 1;
                    }
                }
            }
        }

        if (best_j != -1) {
            path.insert(path.begin() + best_k, best_j);
            is_selected[best_j] = true;
        }
        else {
            break;
        }
    }

    long long Z = calculateObjectiveValue(path);
    return { path, Z };
}

// --- Main Experiment Runner ---

void runExperiment(const string& instance_name) {
    loadInstanceData(instance_name);

    if (N == 0) return;

    ResultStats random_stats, nn_end_stats, nn_all_stats, greedy_cycle_stats;
    const int num_runs_random = 200;
    const int num_runs_greedy = N;

    cout << "\n======================================================\n";
    cout << "--- Running Experiment for " << instance_name << " ---\n";
    cout << "======================================================\n";

    // 1. Random Solutions (200 runs)
    for (int i = 0; i < num_runs_random; i++) {
        auto result = randomSolution();
        updateStats(random_stats, result.first, result.second);
    }

    // 2. Greedy Methods (200 runs starting from each node)
    for (int start_node = 0; start_node < num_runs_greedy; start_node++) {

        auto nn_end_res = nearestNeighborEnd(start_node);
        updateStats(nn_end_stats, nn_end_res.first, nn_end_res.second);

        auto nn_all_res = nearestNeighborAll(start_node);
        updateStats(nn_all_stats, nn_all_res.first, nn_all_res.second);

        auto gc_res = greedyCycle(start_node);
        updateStats(greedy_cycle_stats, gc_res.first, gc_res.second);
    }

    // --- Output Results for Report ---
    auto print_results = [&](const string& name, const ResultStats& stats) {
        if (stats.count == 0) return;
        double avg = (double)stats.sum_Z / stats.count;

        cout << "\n-----------------------------------------\n";
        cout << name << " Results (" << stats.count << " runs)\n";
        cout << "-----------------------------------------\n";
        cout << "MIN Objective Value (Z): " << stats.min_Z << "\n";
        cout << "MAX Objective Value (Z): " << stats.max_Z << "\n";
        cout << "AVG Objective Value (Z): " << fixed << setprecision(2) << avg << "\n";
        cout << "Best Path Indices (K=" << K << "): ";

        for (int i = 0; i < (int)stats.best_path.size(); i++) {
            cout << stats.best_path[i] << (i < stats.best_path.size() - 1 ? ", " : "");
        }
        cout << "\n";
        };

    print_results("Random Solution", random_stats);
    print_results("Nearest Neighbor (End)", nn_end_stats);
    print_results("Nearest Neighbor (All Positions)", nn_all_stats);
    print_results("Greedy Cycle", greedy_cycle_stats);

    // --- VISUALIZATION DATA EXPORT ---
    cout << "\n--- Exporting Visualization Data ---\n";
    exportVisualizationData(instance_name, "Random", random_stats.best_path, random_stats.min_Z);
    exportVisualizationData(instance_name, "NN_End", nn_end_stats.best_path, nn_end_stats.min_Z);
    exportVisualizationData(instance_name, "NN_All", nn_all_stats.best_path, nn_all_stats.min_Z);
    exportVisualizationData(instance_name, "Greedy_Cycle", greedy_cycle_stats.best_path, greedy_cycle_stats.min_Z);
}

// --- Main Function ---
int main() {
    cout << fixed << setprecision(2);

    // Run the experiment for both instances
    runExperiment("TSPA.txt");
    runExperiment("TSPB.txt");

    return 0;
}