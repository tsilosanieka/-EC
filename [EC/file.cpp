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

int calculateDistance(int x1, int y1, int x2, int y2) {
    double dist = sqrt(pow((double)x1 - x2, 2) + pow((double)y1 - y2, 2));
    return static_cast<int>(round(dist));
}

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

long long calculateObjectiveValue(const vector<int>& path) {
    if ((int)path.size() != K) return -1;

    long long total_cost = 0;
    long long total_distance = 0;

    for (int node_id : path) {
        total_cost += all_nodes[node_id].cost;
    }

    for (size_t i = 0; i < path.size(); i++) {
        int from_node = path[i];
        int to_node = path[(i + 1) % path.size()];
        total_distance += DistanceMatrix[from_node][to_node];
    }

    return total_cost + total_distance;
}

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

        while (getline(ss, segment, ';')) {
            try {
                values.push_back(stoi(segment));
            }
            catch (const exception& e) {
                // skip invalid
            }
        }

        if (values.size() >= 3) {
            all_nodes.push_back({ id_counter++, values[0], values[1], values[2] });
        }
    }

    N = all_nodes.size();
    K = (N + 1) / 2;

    if (N == 0) {
        cerr << "ERROR: No node data loaded from file." << endl;
        exit(1);
    }

    initializeDistanceMatrix();
    cout << "Instance " << filename << " loaded. Total nodes (N)=" << N << ", Selected nodes (K)=" << K << endl;
}

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

// --- Heuristics ---

pair<vector<int>, long long> randomSolution() {
    vector<int> all_indices(N);
    iota(all_indices.begin(), all_indices.end(), 0);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine rng(seed);

    shuffle(all_indices.begin(), all_indices.end(), rng);
    vector<int> path(all_indices.begin(), all_indices.begin() + K);
    shuffle(path.begin(), path.end(), rng);

    long long Z = calculateObjectiveValue(path);
    return { path, Z };
}

pair<vector<int>, long long> nearestNeighborEnd(int startNode) {
    vector<int> path = { startNode };
    vector<bool> is_selected(N, false);
    is_selected[startNode] = true;

    while ((int)path.size() < K) {
        int last_node = path.back();
        int best_candidate = -1;
        long long min_delta_Z = numeric_limits<long long>::max();

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

pair<vector<int>, long long> nearestNeighborAll(int startNode) {
    vector<int> path = { startNode };
    vector<bool> is_selected(N, false);
    is_selected[startNode] = true;

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

pair<vector<int>, long long> greedyCycle(int startNode) {
    vector<bool> is_selected(N, false);

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

    for (int i = 0; i < num_runs_random; i++) {
        auto result = randomSolution();
        updateStats(random_stats, result.first, result.second);
    }

    for (int start_node = 0; start_node < num_runs_greedy; start_node++) {
        auto nn_end_res = nearestNeighborEnd(start_node);
        updateStats(nn_end_stats, nn_end_res.first, nn_end_res.second);

        auto nn_all_res = nearestNeighborAll(start_node);
        updateStats(nn_all_stats, nn_all_res.first, nn_all_res.second);

        auto gc_res = greedyCycle(start_node);
        updateStats(greedy_cycle_stats, gc_res.first, gc_res.second);
    }

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
}

// --- Main Function ---

int main() {
    cout << fixed << setprecision(2);

    runExperiment("TSPA.txt");
    runExperiment("TSPB.txt");

    return 0;
}
