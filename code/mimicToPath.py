# Purpose: Construct KG reasoning paths linking patient features to candidate diseases.
import os
from pathlib import Path
import pandas as pd
import cugraph
import pickle
import numpy as np
from tqdm import tqdm
import gc
import sys
import json
import cudf
import rmm
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MIMIC_DIR = DATA_DIR / "mimic"
ENTITY_MATCH_DIR = DATA_DIR / "Entity_Matching"
RESULT_DIR = REPO_ROOT / "results" / "kg_paths"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = RESULT_DIR / "checkpoint.json"
PATIENT_PATHS_BY_DISEASE_FILE = RESULT_DIR / "patient_paths_by_disease.pkl"
PATIENT_PATHS_BY_PATIENT_FILE = RESULT_DIR / "patient_paths_by_patient.pkl"
PATIENT_PATHS_FILE = RESULT_DIR / "patient_paths.pkl"

def load_checkpoint():
    """
    Return {'next_idx': int}; start from 0 if the checkpoint does not exist.
    """
    if CHECKPOINT_FILE.exists():
        with CHECKPOINT_FILE.open("r") as f:
            try:
                data = json.load(f)
                return {"next_idx": int(data.get("next_idx", 0))}
            except Exception:
                # If the checkpoint file is corrupted, restart from 0.
                return {"next_idx": 0}
    return {"next_idx": 0}

def save_checkpoint(next_idx: int):
    """
    Write the checkpoint atomically to avoid partial/corrupted writes.
    """
    tmp = CHECKPOINT_FILE.parent / f"{CHECKPOINT_FILE.name}.tmp"
    with tmp.open("w") as f:
        json.dump({"next_idx": int(next_idx)}, f)
    os.replace(tmp, CHECKPOINT_FILE)
# ------------------------------------------------------

def clear_gpu_cache():
    """
    This function clears the GPU cache to free up memory after each batch of operations.
    """
    # torch.cuda.empty_cache()  # Clear CUDA memory cache (PyTorch)
    rmm.reinitialize()
    gc.collect()  # Python's garbage collection to clean up unused objects
    # print("CUDA cache cleared")

def load_mapping(file_path1, file_path2):
    """
    Loads a mapping from two text files. Each line in the files is assumed to be in the form:
    'query_name: entity_name, entity_index'.
    The mapping will store the tuple (query_name, entity_name) by the entity index,
    skipping entries that have 'NONE' or '-1'.
    """
    mapping = {}

    # Helper function to process each file
    def process_file(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Split the line by ':' to separate query_name and the rest
                    query_name, rest = line.strip().split(':', 1)

                    # Remove square brackets [ and ], and then split by ','
                    rest = rest.strip()[1:-1]  # Remove the square brackets from the list part

                    # Use regex to correctly handle cases where commas are part of the entity name
                    entity_name, entity_index = rest.rsplit(',', 1)  # Split from the right to handle commas in the entity name

                    # Clean the strings by removing extra spaces and quotes
                    entity_name = entity_name.strip().strip('"')
                    query_name = query_name.strip().strip('"')
                    entity_index = int(entity_index.strip())  # Convert entity_index to integer

                    # Skip mappings with 'NONE' or '-1' entity names or entity_index
                    if entity_name == 'NONE' or entity_index == -1:
                        continue

                    # Store query_name and entity_name as a tuple in the mapping
                    mapping[entity_index] = (query_name, entity_name)  # Stripping extra spaces

                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}", flush=True)  # Error handling for malformed lines

    # Process both files
    process_file(file_path1)
    process_file(file_path2)

    return mapping


def find_name_by_index_cu_from_mapping(node_index, mapping):
    """
    Given a node index, return the corresponding query name and entity name from the mapping.
    """
    return mapping.get(node_index, None)  # Return (query_name, entity_name) if found

def find_entity_index_by_query_name(query_name, mapping):
    """
    Given a query name, find the corresponding entity index using the pre-loaded mapping.
    Returns the entity index if found, otherwise None.
    """
    query_name = query_name.strip().strip("'")  # Remove leading/trailing spaces and quotes

    for entity_index, (query, _) in mapping.items():
        query = query.strip().strip("'")  # Clean up the query from the mapping (remove spaces/quotes)
        if query == query_name:
            return entity_index
    return None

def read_hyperedges(file_path):
    hyperedges = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            hyperedges.append([int(i) for i in fields])
    return hyperedges


def find_name_by_index_cu(node_index):
    """
    Get node information including names from a cuDF DataFrame.
    """
    # Find the row where x_index matches the given node_index
    node_info = kg[kg['x_index'] == node_index]

    # Check if the row exists
    if not node_info.empty:
        # Since the result is a cudf DataFrame, we need to use .to_array() to access the values
        name = node_info['x_name'].to_numpy()[0]
        return name

    # Return an empty string if node not found
    return ""

def find_name_by_index_pd(node_index):
    """
    Get node information including names from a pandas DataFrame.
    """
    # Find the row where x_index matches the given node_index
    node_info = kg[kg['x_index'] == node_index]

    # Check if the row exists
    if not node_info.empty:
        # In pandas, you can directly access the value using .iloc
        name = node_info['x_name'].iloc[0]
        return name

    # Return an empty string if node not found
    return ""


def find_shortest_path(kg, source, target, max_depth=3):
    paths = []
    visited = set()
    queue = [(source, [source])]

    pbar = tqdm(total=max_depth, desc="Progress", position=0, leave=True)
    found = False
    stop_depth = max_depth + 1

    while queue and not found:
        current_node, path = queue.pop(0)
        current_depth = len(path)

        if current_depth > pbar.n:
            pbar.update(1)
        pbar.set_postfix({"Time": f"{pbar.format_dict['elapsed']:.2f} sec"})
        pbar.refresh()

        if current_node == target:
            paths.append(path)
            found = True
            stop_depth = current_depth
            continue

        if current_depth < max_depth and current_depth <= stop_depth:
            for neighbor in get_neighbors(kg, current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    pbar.close()
    return paths


def sample_graph_edges(graph, fraction=0.7):
    """
    Sample a fraction of the edges from the graph to reduce its size.
    """
    num_edges = graph.number_of_edges()
    sample_size = int(num_edges * fraction)

    # Sample edges randomly
    sampled_edges = graph.edges.sample(sample_size)

    # Create a subgraph from the sampled edges
    subgraph = cugraph.Graph(directed=True)
    subgraph.from_cudf_edgelist(sampled_edges, source='src', destination='dst')

    return subgraph



def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    needles = ["out of memory", "std::bad_alloc", "cuda error", "rmm", "failed to allocate"]
    return any(n in msg for n in needles)

def find_shortest_path_cugraph(graph, source, target, depth_limit=3):
    """
    Perform a breadth-first search (BFS) up to a certain depth (depth_limit)
    from the source node and reconstruct all shortest paths from source to target.
    This function retries with a subgraph if OOM occurs and reduces depth if OOM still happens.
    """
    try:
        # First, try running BFS with the original graph and the given depth limit
        result_df = cugraph.bfs(graph, source, depth_limit=depth_limit)
        df = result_df[['vertex', 'distance', 'predecessor']].copy()
        del result_df  # free BFS result memory
        reached_val = df['vertex'].eq(target).sum()
        try:
            # print("Reached\n")
            reached = int(reached_val)            # may already be numpy.int64
        except Exception:
            reached = int(reached_val.get())      # older builds return device scalar
        if reached == 0:
            del df
            return []
        # # Convert cuDF Series to a Python list of vertices
        # vertex_list = df['vertex'].to_arrow().to_pylist()

        # # Check if the target is reachable
        # if target not in vertex_list:
        #     del df
        #     return []

        # Initialize list to hold all possible paths
        all_paths = []

        # Recursive function to find all paths from source to target
        def find_all_paths(current_node, path):
            if current_node == source:
                all_paths.append(path[::-1])  # Add the path in reverse order
                return

            # Get the predecessor of the current node
            predecessor = df[df['vertex'] == current_node]['predecessor'].to_numpy()[0]
            if predecessor == -1:
                return  # No predecessor, can't go further back

            # Add the predecessor to the current path and recurse
            find_all_paths(predecessor, path + [predecessor])

        # Start path reconstruction from the target node
        find_all_paths(target, [target])

        all_paths = [[int(node) for node in path] for path in all_paths]

        # Clear the GPU cache after the BFS operation
        clear_gpu_cache()
        del df

        return all_paths

    except Exception as e:
        # Try to clean GPU/RMM resources before exit to reduce cuFile assertion risk.
        try:
            clear_gpu_cache()
        except Exception:
            pass
        print(f"[OOM] BFS(depth={depth_limit}) failed: {e}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        os._exit(137)   # Non-zero exit code to trigger Slurm retry when requeue is enabled




def get_neighbors(kg, node):
    neighbors = kg[kg['x_index'] == node]['y_index'].values.tolist()
    return neighbors



def record_relation(node1_index, node2_index):
    """
    Record the relation between two nodes as (node1, node2): relation.
    """
    # Find the relation between two nodes
    relations = kg[(kg['x_index'] == node1_index) & (kg['y_index'] == node2_index)]
    if relations.empty:
        relations = kg[(kg['x_index'] == node2_index) & (kg['y_index'] == node1_index)]

    # Assume one relation is sufficient for this pair
    relation_name = relations['relation'].to_numpy()[0] if not relations.empty else ""

    # Store relation in dictionary as (node1, node2): relation
    node_relation_dict[(node1_index, node2_index)] = relation_name



def get_path_details(path, mapping):
    """
    Given a path (list of node indices), reconstruct a detailed string with nodes and relations.
    The first and last node names are replaced with the original query names, and middle nodes are replaced with entity names.
    Format: "{node1}(node)->{edge1}(edge)->{node2}(node)->..."
    """
    path_details = []

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]

        # Get node names based on indices (replace with original names)
        if i == 0:  # First node (head) gets query_name
            node1_name, _ = find_name_by_index_cu_from_mapping(node1, mapping)
        else:  # Middle nodes get entity_name
            node1_name = find_name_by_index_pd(node1)

        if i == len(path) - 2:  # Last node gets query_name
            node2_name, _ = find_name_by_index_cu_from_mapping(node2, mapping)
        else:  # Middle nodes get entity_name
            node2_name = find_name_by_index_pd(node2)

        # Record the relationship between nodes
        record_relation(node1, node2)

        # Get the relationship (edge) between the nodes
        relation = node_relation_dict.get((node1, node2), "")

        # Add the first node info (with "node" label)
        if i == 0:
            path_details.append(f"{node1_name}(node)")

        # Add the edge relation and second node info
        path_details.append(f"->{relation}(edge)->{node2_name}(node)")

    return ''.join(path_details)


def process_patient_paths(patient_idx, edge_label, hyperedges, edge_list, node_text, graph, mapping):
    """
    Process the paths for a given patient and store the results in patient_paths.
    Each patient ID is explicitly included in the paths.
    """
    patient_paths = {}  # Initialize the patient paths dictionary

    # Iterate over all edge labels (whether 0 or 1)
    for feature_idx, label in enumerate(edge_label):
        # Get disease/feature name from the edge_list (mapping index to disease/feature)
        disease_name = edge_list[str(feature_idx)]
        # Find the corresponding entity index for the disease/feature
        end_index = find_entity_index_by_query_name(disease_name, mapping)

        if end_index is not None:
            # Initialize the structure for storing paths and label for this disease
            if patient_idx not in patient_paths:
                patient_paths[patient_idx] = {}

            if disease_name not in patient_paths[patient_idx]:
                patient_paths[patient_idx][disease_name] = {
                    "paths": [],  # List of paths
                    "label": label  # Single label per disease
                }

            # Search paths for the feature, regardless of the label
            feature_index_list = hyperedges[patient_idx]  # Get the feature index list for this patient

            start_indices = []
            for feature in feature_index_list:
                feature_name = node_text[str(feature)]
                entity_index = find_entity_index_by_query_name(feature_name, mapping)
                if entity_index is not None:
                    start_indices.append(entity_index)

            # Find all paths between start and end indices (from patient data)
            for start in start_indices:
                # For each valid (start, end), perform path search
                all_paths = find_shortest_path_cugraph(graph, start, end_index)  # Search for paths
                for path in all_paths:
                    # Create a path string with the correct format
                    path_details = get_path_details(path, mapping)

                    # Append the path to the list of paths for this disease
                    patient_paths[patient_idx][disease_name]["paths"].append(path_details)

    return patient_paths


def save_patient_paths_by_disease(patient_paths):
    """
    Save all the patient paths into one single file, categorized by disease and patient.
    The final structure of the file will look like:
    {
        disease_name: {
            patient_id: {
                "paths": ["path1", "path2", ...],
                "label": label
            }
        }
    }
    """
    # Initialize the structure for storing paths by disease and patient
    disease_patient_paths = {}

    # If the file already exists, load the existing patient paths
    if os.path.exists(PATIENT_PATHS_BY_DISEASE_FILE):
        with open(PATIENT_PATHS_BY_DISEASE_FILE, "rb") as f:
            disease_patient_paths = pickle.load(f)

    # Iterate over each patient and disease to organize paths by disease and patient
    for patient_id, diseases in patient_paths.items():
        for disease_name, path_data in diseases.items():
            # Only add disease data if there are paths
        # if path_data["paths"]:
            # If disease_name doesn't exist in disease_patient_paths, initialize it
            if disease_name not in disease_patient_paths:
                disease_patient_paths[disease_name] = {}

            # Ensure that we initialize the patient's path list and labels list if not already present
            if patient_id not in disease_patient_paths[disease_name]:
                disease_patient_paths[disease_name][patient_id] = {
                    "paths": [],
                    "label": path_data["label"]  # Only one label per disease
                }

            # Store the paths for each patient under the disease_name
            for path in path_data["paths"]:
                disease_patient_paths[disease_name][patient_id]["paths"].append(path)

    # Save the reorganized patient paths by disease_name into a single file
    with open(PATIENT_PATHS_BY_DISEASE_FILE, "wb") as f:
        pickle.dump(disease_patient_paths, f)

    print("Saved all patient paths to 'patient_paths_by_disease.pkl'.", flush=True)


def save_patient_paths_by_patient(patient_paths):
    """
    Save the patient paths in the desired format for calculate_statistics.
    The structure will be:
    {
        patient_id: {
            disease_name: {
                "paths": ["path1", "path2", ...],
                "label": label
            }
        }
    }
    """
    # Initialize a dictionary to hold the paths in the desired structure
    formatted_patient_paths = {}

    # Iterate through the original patient_paths and reformat
    for patient_id, diseases in patient_paths.items():
        for disease_name, path_data in diseases.items():
            # Even if there are no paths, add the disease with empty paths
            if patient_id not in formatted_patient_paths:
                formatted_patient_paths[patient_id] = {}  # Initialize dictionary for the patient

            if disease_name not in formatted_patient_paths[patient_id]:
                formatted_patient_paths[patient_id][disease_name] = {
                    "paths": [],  # Empty paths will be included
                    "label": path_data["label"]  # Only one label per disease
                }

            # Append the paths (even if empty) to the correct disease for the patient
            for path in path_data["paths"]:
                formatted_patient_paths[patient_id][disease_name]["paths"].append(path)

    # Save all the patient paths in one large file
    with open(PATIENT_PATHS_BY_PATIENT_FILE, "wb") as f:
        pickle.dump(formatted_patient_paths, f)

    print("All patient paths saved in the 'patient_paths_by_patient.pkl' file.", flush=True)


def load_patient_paths_by_disease():
    """
    Load all the paths from one file, categorized by disease and patient.
    Returns a dictionary of the structure:
    {
        disease_name: {
            patient_id: {
                "paths": ["path1", "path2", ...],
                "label": label
            }
        }
    }
    """
    patient_paths = {}

    # Load all the patient paths from one single file
    with open(PATIENT_PATHS_BY_DISEASE_FILE, "rb") as f:
        patient_paths = pickle.load(f)

    return patient_paths

def load_patient_paths_by_patient():
    """
    Load the patient paths in the desired format for calculating statistics.
    The structure will be:
    {
        patient_id: {
            disease_name: {
                "paths": ["path1", "path2", ...],
                "label": label
            }
        }
    }
    """
    # Load the saved patient paths from the file where paths are categorized by patient
    with open(PATIENT_PATHS_BY_PATIENT_FILE, "rb") as f:
        formatted_patient_paths = pickle.load(f)

    return formatted_patient_paths

def calculate_path_length(path):
    """
    Calculate the length of a path by counting the number of edges (occurrences of "(edge)") in the path string.
    """
    # Count occurrences of "(edge)" in the path string
    edge_count = path.count("(edge)")

    return edge_count

def calculate_statistics(patient_paths, edge_labels, edge_list, node_text, mapping):
    """
    Calculate and display statistics for the paths.
    """
    total_patients = len(patient_paths)
    total_endpoints = 0
    total_paths = 0
    total_nodes_used = 0
    total_paths_per_endpoint = 0
    total_path_length = 0
    endpoint_counts = {}
    feature_statistics = {}

    for patient_id, paths in patient_paths.items():
        for disease_name, path_data in paths.items():
            total_endpoints += 1
            num_paths = len(path_data["paths"])
            total_paths += num_paths
            total_paths_per_endpoint += num_paths
            total_path_length += sum(calculate_path_length(path) for path in path_data["paths"])

            # Tracking statistics for each feature-label match
            features_used = edge_labels[patient_id]
            non_zero_features = np.where(features_used == 1)[0]
            total_nodes_used += len(non_zero_features)

            for feature in non_zero_features:
                disease_name = edge_list[str(feature)]
                feature_index = find_entity_index_by_query_name(disease_name, mapping)
                if feature_index is None:
                    continue
                if feature_index not in feature_statistics:
                    feature_statistics[feature_index] = {
                        "num_paths": 0,
                        "total_length": 0
                    }

                feature_statistics[feature_index]["num_paths"] += num_paths
                feature_statistics[feature_index]["total_length"] += sum(calculate_path_length(path) for path in path_data["paths"])

    # Compute overall statistics
    avg_paths_per_patient = total_paths / total_patients if total_patients else 0
    avg_path_length = total_path_length / total_paths if total_paths else 0
    avg_paths_per_endpoint = total_paths_per_endpoint / total_endpoints if total_endpoints else 0
    avg_features_used_per_patient = total_nodes_used / total_patients if total_patients else 0

    print("\nOverall Statistics:")
    print(f"Average number of paths per patient: {avg_paths_per_patient}")
    print(f"Average path length per path: {avg_path_length}")
    print(f"Average number of paths per endpoint: {avg_paths_per_endpoint}")
    print(f"Average features used per patient: {avg_features_used_per_patient}")

    print("\nFeature-Label Statistics:")
    for feature_index, stats in feature_statistics.items():
        avg_paths_per_feature = stats["num_paths"] / total_patients if total_patients else 0
        avg_length_per_feature = stats["total_length"] / stats["num_paths"] if stats["num_paths"] else 0
        print(f"Feature {feature_index}:")
        print(f"  Average number of paths: {avg_paths_per_feature}")
        print(f"  Average path length: {avg_length_per_feature}")

def find_nonzero_features_and_paths(edge_labels, edge_list, hyperedges, node_text, graph, mapping):
    ckpt = load_checkpoint()
    start_idx = int(ckpt.get("next_idx", 0))
    total = len(edge_labels)
    if start_idx >= total:
        print(f"Checkpoint already at end (next_idx={start_idx}); moving to statistics stage.", flush=True)
    else:
        print(f"Resuming from checkpoint: patient_idx = {start_idx}/{total}", flush=True)

        # Use range + initial=start_idx so tqdm shows resumed progress correctly
    for patient_idx in tqdm(range(start_idx, total),
                            total=total,
                            desc="Processing Patients",
                            ncols=100,
                            initial=start_idx):
        try:
            edge_label = edge_labels[patient_idx]

            # Process only the current patient to keep memory usage bounded
            patient_paths_one = process_patient_paths(
                patient_idx, edge_label, hyperedges, edge_list, node_text, graph, mapping
            )

            # Merge into on-disk files (dedup happens internally)
            save_patient_paths_by_disease(patient_paths_one)
            save_patient_paths_by_patient(patient_paths_one)

            # Advance checkpoint immediately after successful write
            save_checkpoint(patient_idx + 1)
            print(f"Saved patient {patient_idx} successfully", flush=True)

            # Clean up aggressively if memory pressure is a concern
            del patient_paths_one
            gc.collect()
            clear_gpu_cache()  # Clear CUDA memory

        except Exception as e:
            # Treat any exception like OOM: roll back to current patient for retry
            print(f"[ERROR] Failed to process patient {patient_idx}: {e}", flush=True)
            save_checkpoint(patient_idx)      # Keep checkpoint on current patient for retry
            try:
                clear_gpu_cache()
            except Exception:
                pass
            sys.stdout.flush(); sys.stderr.flush()
            time.sleep(0.5)
            os._exit(137)
    loaded_patient_paths = load_patient_paths_by_patient()
    calculate_statistics(loaded_patient_paths, edge_labels, edge_list, node_text, mapping)

    # Optional: remove checkpoint after full completion
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("All samples processed; checkpoint removed.", flush=True)
    except Exception:
        pass

    return  # Avoid returning large in-memory dictionaries



def load_saved_paths():
    """
    Load previously saved path data and continue reading from disk.
    """
    if os.path.exists(PATIENT_PATHS_FILE):
        with open(PATIENT_PATHS_FILE, "rb") as f:
            paths = []
            try:
                while True:
                    paths.append(pickle.load(f))
            except EOFError:
                pass
            return paths
    else:
        return []


def main():
    # Main entry
    mapping_file1 = ENTITY_MATCH_DIR / "existing_nodes_manually_refined.txt"
    mapping_file2 = ENTITY_MATCH_DIR / "existing_edges_manually_refined.txt"
    mapping = load_mapping(mapping_file1, mapping_file2)
    patient_paths = find_nonzero_features_and_paths(edge_labels, edge_list, hyperedges, node_text, graph, mapping)


edge_labels = pd.read_csv(MIMIC_DIR / "edge-labels-mimic3_updated_truncated.txt", header=None).values
with open(MIMIC_DIR / "edge_text.json", "r") as f:
    edge_list = json.load(f)

hyperedges = read_hyperedges(MIMIC_DIR / "hyperedges-mimic3_truncated.txt")

with open(MIMIC_DIR / "node_text.json", "r") as f:
    node_text = json.load(f)


# Load full KG data
kg_cudf = cudf.read_csv(DATA_DIR / "prune_kg.csv")
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(kg_cudf, source='x_index', destination='y_index', renumber=False)

# Move KG data to CPU for name/relation lookups
kg = kg_cudf.to_pandas()
del kg_cudf  # free GPU memory for KG DataFrame


# kg = pd.read_csv('../data/prune_kg.csv')
# graph = nx.DiGraph()  # For a directed graph, use DiGraph, otherwise use Graph() for undirected
#
# # Step 3: Add edges to the graph
# for index, row in kg.iterrows():
#     graph.add_edge(row['x_index'], row['y_index'])


# Save full graph (optional)
# graph.to_pickle('graph.pkl')

# Cache computed node-endpoint paths
computed_paths = {}


# Global dictionary to store relations between nodes
node_relation_dict = {}

# batch_size = 100
if __name__ == '__main__':
    main()
