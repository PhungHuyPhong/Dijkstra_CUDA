#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <device_atomic_functions.h>
#define N 10000 // so đỉnh
#define THREADS_PER_BLOCK 256

__global__ void findMinDistanceKernel(int* distances, bool* visited, int* minDist, int* minIndex, int V) {
    __shared__ int sharedMinDist[THREADS_PER_BLOCK];
    __shared__ int sharedMinIndex[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    sharedMinDist[localTid] = INT_MAX;
    sharedMinIndex[localTid] = -1;

    if (tid < V && !visited[tid]) {
        sharedMinDist[localTid] = distances[tid];
        sharedMinIndex[localTid] = tid;
    }
    __syncthreads();
    // Song song hóa giảm thiểu (Reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localTid < stride) {
            if (sharedMinDist[localTid] > sharedMinDist[localTid + stride]) {
                sharedMinDist[localTid] = sharedMinDist[localTid + stride];
                sharedMinIndex[localTid] = sharedMinIndex[localTid + stride];
            }
        }
    __syncthreads();
    }
    if (localTid == 0) {
        atomicMin(minDist, sharedMinDist[0]);
        if (*minDist == sharedMinDist[0]) {
            *minIndex = sharedMinIndex[0];
        }
    }
}

__global__ void relaxEdgesKernel(int* adjMatrix, int* distances, bool* visited, int u, int V) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < V && !visited[tid] && adjMatrix[u * V + tid] > 0) {
        int newDist = distances[u] + adjMatrix[u * V + tid];
        if (newDist < distances[tid]) {
            distances[tid] = newDist;
        }
    }
}

void dijkstraSequential(int* adjMatrix, int* distances, int V, int src) {
    bool* visited = (bool*)malloc(V * sizeof(bool));

    for (int i = 0; i < V; i++) {
        distances[i] = INT_MAX;
        visited[i] = false;
    }
    distances[src] = 0;

    for (int i = 0; i < V - 1; i++) {
        int minDist = INT_MAX, minIndex = -1;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && distances[v] < minDist) {
                minDist = distances[v];
                minIndex = v;
            }
        }

        if (minIndex == -1) break;

        visited[minIndex] = true;

        for (int v = 0; v < V; v++) {
            if (!visited[v] && adjMatrix[minIndex * V + v] && distances[minIndex] != INT_MAX &&
                distances[minIndex] + adjMatrix[minIndex * V + v] < distances[v]) {
                distances[v] = distances[minIndex] + adjMatrix[minIndex * V + v];
            }
        }
    }

    free(visited);
}

void dijkstraParallel(int* adjMatrix, int* distances, int V, int src) {
    int* d_adjMatrix, * d_distances;
    bool* d_visited;
    int* d_minDist, * d_minIndex;

    cudaMalloc(&d_adjMatrix, V * V * sizeof(int));
    cudaMalloc(&d_distances, V * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(bool));
    cudaMalloc(&d_minDist, sizeof(int));
    cudaMalloc(&d_minIndex, sizeof(int));

    
    cudaMemcpy(d_adjMatrix, adjMatrix, V * V * sizeof(int), cudaMemcpyHostToDevice);
    

    bool* visited = (bool*)malloc(V * sizeof(bool));
    for (int i = 0; i < V; i++) {
		distances[i] = INT_MAX;
		visited[i] = false;
    }
    distances[src] = 0;
    cudaMemcpy(d_distances, distances, V * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < V - 1; i++) {
        int minDist = INT_MAX, minIndex = -1;
        cudaMemcpy(d_visited, visited, V * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_minDist, &minDist, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_minIndex, &minIndex, sizeof(int), cudaMemcpyHostToDevice);

        int numBlocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        findMinDistanceKernel << <numBlocks, THREADS_PER_BLOCK >> > (d_distances, d_visited, d_minDist, d_minIndex, V);

        cudaMemcpy(&minDist, d_minDist, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);

        if (minIndex == -1) break;

        visited[minIndex] = true;
        cudaMemcpy(d_visited, visited, V * sizeof(bool), cudaMemcpyHostToDevice);

        relaxEdgesKernel << <numBlocks, THREADS_PER_BLOCK >> > (d_adjMatrix, d_distances, d_visited, minIndex, V);
    }

    cudaMemcpy(distances, d_distances, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_adjMatrix);
    cudaFree(d_distances);
    cudaFree(d_visited);
    cudaFree(d_minDist);
    cudaFree(d_minIndex);
    free(visited);
}

int main() {
    int V = N;
    int* adjMatrix = (int*)malloc(V * V * sizeof(int));
    int* distancesSerial = (int*)malloc(V * sizeof(int));
    int* distancesParallel = (int*)malloc(V * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i != j) {
                adjMatrix[i * V + j] = rand() % 10 + 1; // Trọng số từ 1 đến 10
            }
            else {
                adjMatrix[i * V + j] = 0; // Đường chéo chính là 0
            }
        }
    }

    clock_t start = clock();
    dijkstraSequential(adjMatrix, distancesSerial, V, 0);
    clock_t end = clock();
    printf("Serial Dijkstra time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    dijkstraParallel(adjMatrix, distancesParallel, V, 0);
    end = clock();
    printf("Parallel Dijkstra time: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

	for (int i = 0; i < V; i++) {
		if (distancesSerial[i] != distancesParallel[i]) {
			printf("Results are different\n");
			break;
		}
	}

    free(adjMatrix);
    free(distancesSerial);
    free(distancesParallel);

    return 0;
}

