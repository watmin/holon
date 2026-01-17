# Holon Architecture

## Data Flow Overview

Holon's architecture centers around transforming structured data into high-dimensional vector representations for efficient similarity-based operations. The system supports ingestion of JSON and EDN data blobs, encoding them via VSA/HDC principles, and querying for similar data.

### Ingestion Flow

1. **Data Input**: Accept a data blob as a JSON or EDN string.
2. **Parsing**: Convert string to internal data structure (dict for JSON, appropriate for EDN).
3. **Recursive Encoding**: Encode the structure preserving relationships:
   - **Maps**: Bind keys to values using element-wise multiplication.
   - **Sequences**: Bundle encoded items.
   - **Sets**: Bundle items with a set indicator.
   - **Scalars**: Map to atomic vectors.
4. **Vector Allocation**: For each unique atom/relationship, allocate or retrieve a 16k-dimensional vector with bipolar values {-1, 0, 1}.
5. **Structural Binding/Bundling**: Combine vectors using binding (for relationships) and bundling (for aggregation).
6. **Storage**: Store the encoded vector along with the original data blob in the chosen backend (initially in-memory).

### Query Flow

1. **Probe Input**: Accept a query probe (JSON/EDN string or partial data).
2. **Atomization**: Same as ingestion – break probe into atoms.
3. **Vector Encoding**: Encode probe into a vector using the same binding/bundling operations.
4. **Similarity Computation**: Compute cosine similarity (or other metrics) between probe vector and all stored data blob vectors.
5. **Retrieval**: Return top-matching data blobs with their similarity scores, above a configurable threshold.

### Backend Abstraction

- **Local CPU**: All operations in memory using NumPy or similar for vector math.
- **Local GPU**: Use CuPy or PyTorch for GPU acceleration.
- **Remote**: HTTP client interfaces with a service that uses MongoDB for data storage and Qdrant for vector search.

The abstract `Store` interface hides backend details, allowing seamless switching between modes.

## Key Components

- **Atomizer**: Handles parsing and atom extraction from JSON/EDN.
- **VectorManager**: Manages atom-to-vector mappings and allocation.
- **Encoder**: Performs binding and bundling operations.
- **SimilarityEngine**: Computes vector similarities.
- **Store Interface**: Abstract base class for backends (CPU, GPU, Remote).

## Design Considerations

- **Dimensionality**: Start with 16k dimensions; configurable for performance tuning.
- **Vector Values**: Bipolar {-1, 0, 1} for robustness in HDC operations.
- **Scalability**: Initial focus on in-memory CPU; future backends for larger scales.
- **Similarity Threshold**: User-configurable for query precision/recall trade-offs.