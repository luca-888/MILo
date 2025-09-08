from typing import Tuple
import torch
from tqdm import tqdm


def compute_adjacency_from_tets_no_chunking(delaunay_tets: torch.Tensor, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the adjacency information from Delaunay tetrahedra.
    This function processes a set of Delaunay tetrahedra to compute the adjacency
    list and offsets for each point in the tetrahedra.

    Args:
        delaunay_tets (torch.Tensor): A tensor of shape (N, 4) representing the indices
                                      of points forming the tetrahedra.
        num_points (int): The index of the last point in the tetrahedra.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - point_adjacency (torch.Tensor): A tensor containing the adjacency list of points.
            - point_adjacency_offsets (torch.Tensor): A tensor containing the offsets for each point
                                                      in the adjacency list.
    """
    num_tets = delaunay_tets.size(0)
    x1_chunk = delaunay_tets.repeat(1, 4).view(-1)
    x2_chunk = delaunay_tets.repeat_interleave(4, dim=1).view(-1)

    # Mask out self-edges
    mask_chunk = x1_chunk != x2_chunk
    edges = torch.stack([x1_chunk[mask_chunk], x2_chunk[mask_chunk]], dim=1)

    # Sort the edges by the source point
    src, dst = edges[:, 0], edges[:, 1]
    sorted_src, sorted_indices = torch.sort(src)
    sorted_dst = dst[sorted_indices]

    # After sorting, we have two tensors:
    # - sorted_src: This tensor contains the source points, e.g., [0, 0, 0, 0, 1, 1, ...].
    # - sorted_dst: This tensor contains the destination points corresponding to each source point, e.g., [7, 4, 9, 1, 6, 8, ...].
    # Meaning:
    #   - The point '0' is connected to the points [7, 4, 9, 1].
    #   - The point '1' is connected to the points [6, 8, ...].

    # We collapse the edges by counting how many edges per point
    counts = torch.bincount(sorted_src, minlength=num_points)

    point_adjacency_offsets = torch.zeros(num_points + 1, device=edges.device, dtype=torch.uint32)
    # The cumsum gives us the offset for each point
    point_adjacency_offsets[1:] = torch.cumsum(counts, dim=0)
    point_adjacency = sorted_dst
    return point_adjacency, point_adjacency_offsets


def compute_adjacency_from_tets(delaunay_tets: torch.Tensor, num_points: int, chunk_size: int = 5_000_000, index_dtype: torch.dtype = torch.long, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tets = delaunay_tets.size(0)
    # Calculate the number of chunks
    num_chunks = (num_tets + chunk_size - 1) // chunk_size

    all_point_adjacency_offsets = []
    all_point_adjacency = []
    for i in tqdm(range(num_chunks), desc="Computing adjacency list"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_tets)
        chunk_tets = delaunay_tets[start_idx:end_idx].to(device)

        # Get all edges in the chunk
        x1_chunk = chunk_tets.repeat(1, 4).view(-1)
        x2_chunk = chunk_tets.repeat_interleave(4, dim=1).view(-1)

        # Mask out self-edges
        mask_chunk = x1_chunk != x2_chunk
        edges_chunk = torch.stack([x1_chunk[mask_chunk], x2_chunk[mask_chunk]], dim=1)

        # Sort the edges by the source point
        src, dst = edges_chunk[:, 0], edges_chunk[:, 1]
        sorted_src, sorted_indices = torch.sort(src)
        sorted_dst = dst[sorted_indices]

        # After sorting, we have two tensors:
        # - sorted_src: This tensor contains the source points, e.g., [0, 0, 0, 0, 1, 1, ...].
        # - sorted_dst: This tensor contains the destination points corresponding to each source point, e.g., [7, 4, 9, 1, 6, 8, ...].
        # Meaning:
        #   - The point '0' is connected to the points [7, 4, 9, 1].
        #   - The point '1' is connected to the points [6, 8, ...].

        # Count the number of edges per point
        counts = torch.bincount(sorted_src, minlength=num_points)

        # Build the offsets for each point
        point_adjacency_offsets_for_chunk = torch.zeros(num_points + 1, dtype=index_dtype)
        point_adjacency_offsets_for_chunk[1:] = torch.cumsum(counts, dim=0).cpu()
        point_adjacency_for_chunk = sorted_dst.cpu()

        all_point_adjacency_offsets.append(point_adjacency_offsets_for_chunk)
        all_point_adjacency.append(point_adjacency_for_chunk)

    # Merging all_point_adjacency_offsets
    point_adjacency_offsets = torch.vstack(all_point_adjacency_offsets).sum(dim=0).to(device)

    # Merging all_point_adjacency
    point_adjacency = torch.zeros(point_adjacency_offsets[-1], dtype=index_dtype, device=device)

    for i in tqdm(range(num_chunks), desc="Merging adjacency list"):
        chunk_offset = all_point_adjacency_offsets[i].to(device)
        n_edges_in_chunk = chunk_offset[1:] - chunk_offset[:-1]

        range_starts = point_adjacency_offsets[:-1]
        total_len = all_point_adjacency[i].shape[0]

        # Step 1: repeat each start[i] for lengths[i] times
        range_starts_repeated = torch.repeat_interleave(range_starts, n_edges_in_chunk)

        cumsum_n_edges = torch.cumsum(torch.cat([torch.tensor([0], dtype=index_dtype, device=device), 
                                                 n_edges_in_chunk[:-1]]), 0)
        repeated_cumsum_n_edges = torch.repeat_interleave(cumsum_n_edges, n_edges_in_chunk)
        # Step 2: build offsets per group
        offsets = torch.arange(total_len, dtype=index_dtype, device=device) - repeated_cumsum_n_edges

        # Final indices: start + offset
        indices = range_starts_repeated + offsets

        point_adjacency.index_add_(0, 
                                index=indices.to(index_dtype), 
                                source=all_point_adjacency[i].to(index_dtype).to(device))

    return point_adjacency, point_adjacency_offsets




