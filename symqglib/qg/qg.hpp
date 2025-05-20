#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/time.h>

#include "../common.hpp"
#include "../quantization/rabitq.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"

// #define DEBUG

namespace symqg {
/**
 * @brief this Factor only for illustration, the true storage is continous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0
    float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
    float factor_vq;  // Factor of v_l * ||q_r||
};

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dimension_ = 0;     // dimension
    size_t padded_dim_ = 0;    // padded dimension
    PID entry_point_ = 0;      // Entry point of graph

    mutable size_t scan_count_;
    mutable size_t scanner_pop_time_, scanner_scan_time_, scanner_l2_sqr_time_, scanner_insert_result_time_;
    mutable size_t collector_insert_time_;
    mutable size_t num_collector_try_insert_, num_collector_insert_;
    mutable size_t collector_insert_branch_time_;

    mutable size_t num_is_in_head_;

    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<
            float,
            1 << 22,
            true>>
        data_;  // vectors + graph + quantization codes
    QGScanner scanner_;
    FHTRotator rotator_;
    HashBasedBooleanSet visited_;
    buffer::SearchBuffer search_pool_;

    /*
     * Position of different data in each row
     *      RawData + QuantizationCodes + Factors + neighborIDs
     * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
     * we do not need to store the degree for each vertex
     */
    size_t code_offset_ = 0;      // pos of packed code
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    size_t row_offset_ = 0;       // length of entire row

    void initialize();

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void copy_vectors(const float*);

    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(&data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] const uint8_t* get_packed_code(PID data_id) const {
        return reinterpret_cast<const uint8_t*>(
            &data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void
    find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float*);

    float scan_neighbors(
        const QGQuery& q_obj,
        const float* cur_data,
        float* appro_dist,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree
    ) const;

   public:
    explicit QuantizedGraph(size_t, size_t, size_t);

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dimension_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save_index(const char*) const;

    void load_index(const char*);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_(1 << ceil_log2(dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(dimension_)
    , visited_(100)
    , search_pool_(0) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_.save(output);

    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::cout << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    /* Check file size */
    size_t filesize = get_filesize(filename);
    size_t correct_size = sizeof(PID) + (sizeof(float) * num_points_ * row_offset_) +
                          (sizeof(float) * padded_dim_);
    if (filesize != correct_size) {
        std::cerr << "Index file size error! Please make sure the index and "
                     "init parameters are correct\n";
        abort();
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_.load(input);

    input.close();
    std::cout << "Quantized graph loaded!\n";
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->search_pool_.resize(cur_ef);
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

/*
 * search single query
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    /* Init query matrix */
    this->visited_.clear();
    this->search_pool_.clear();

    scan_count_ = 0;
    scanner_l2_sqr_time_ = 0;
    scanner_scan_time_ = 0;
    scanner_pop_time_ = 0;
    scanner_insert_result_time_ = 0;
    collector_insert_time_ = 0;
    num_collector_try_insert_ = 0;
    num_collector_insert_ = 0;
    collector_insert_branch_time_ = 0;

#if defined(DEBUG)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    search_qg(query, knn, results);
#if defined(DEBUG)
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "[master] total time:              " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() << " ns\t" << scan_count_ << std::endl;
    std::cout << "[master] num_scann, num_collect:  " << scan_count_ << ", " << scan_count_ << std::endl;
    std::cout << "[scanner] pop:                    " << scanner_pop_time_ << " ns\t" << scanner_pop_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
    std::cout << "[scanner] l2_sqr:                 " << scanner_l2_sqr_time_ << " ns\t" << scanner_l2_sqr_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
    std::cout << "[scanner] scan:                   " << scanner_scan_time_ << " ns\t" << scanner_scan_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
    std::cout << "[scanner] insert result:          " << scanner_insert_result_time_ << " ns\t" << scanner_insert_result_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
    std::cout << "[collector] insert:               " << collector_insert_time_ << " ns\t" << collector_insert_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
    std::cout << "[collector] num_insert(try):      " << (float)num_collector_insert_ / num_collector_try_insert_ << "\t\t" << num_collector_insert_ << '(' << num_collector_try_insert_<< ')' << std::endl;
#endif
    std::cout << "[collector] insert branch:       " << collector_insert_branch_time_ << " ns\t" << collector_insert_branch_time_ / scan_count_ << " ns\t" << scan_count_ << std::endl;
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   searh res
 */
inline void QuantizedGraph::search_qg(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    // query preparation
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    search_pool_.insert(this->entry_point_, FLT_MAX);

    /* Result pool */
    buffer::ResultBuffer res_pool(knn);

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis

    // PID loop_node = 0;
    // for (; loop_node < num_points_; loop_node++) {
    //     float sqr_y = scan_neighbors(
    //         q_obj,
    //         get_vector(loop_node),
    //         appro_dist.data(),
    //         this->search_pool_,
    //         this->degree_bound_
    //     );
    //     res_pool.insert(loop_node, sqr_y);
    //     if (loop_node == num_points_ - 1) {
    //         loop_node = 0;
    //     }
    // }

    while (search_pool_.has_next()) {
#if defined(DEBUG)
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        PID cur_node = search_pool_.pop();
        if (visited_.get(cur_node)) {
#if defined(DEBUG)
            auto t2 = std::chrono::high_resolution_clock::now();
            scanner_pop_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
#endif
            continue;
        }
        scan_count_++;
        visited_.set(cur_node);
        const float* cur_data = get_vector(cur_node);

#if defined(DEBUG)
        auto t2 = std::chrono::high_resolution_clock::now();
        scanner_pop_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
#endif
        volatile float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

#if defined(DEBUG)
        t1 = std::chrono::high_resolution_clock::now();
        scanner_l2_sqr_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t2).count();
#endif

        res_pool.insert(cur_node, sqr_y);

#if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        scanner_insert_result_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
#endif

        /* Compute approximate distance by Fast Scan */
        const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
        const auto* factor = &cur_data[factor_offset_];
        this->scanner_.scan_neighbors(
            appro_dist.data(),
            q_obj.lut().data(),
            sqr_y,
            q_obj.lower_val(),
            q_obj.width(),
            q_obj.sumq(),
            packed_code,
            factor
        );

#if defined(DEBUG)
        t1 = std::chrono::high_resolution_clock::now();
        scanner_scan_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t2).count();
#endif

        const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
        for (uint32_t i = 0; i < degree_bound_; ++i) {
            num_collector_try_insert_++;
            PID cur_neighbor = ptr_nb[i];
            float tmp_dist = appro_dist[i];
            auto t1 = std::chrono::high_resolution_clock::now();
            if (search_pool_.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
                auto t2 = std::chrono::high_resolution_clock::now();
                collector_insert_branch_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                continue;
            }
            num_collector_insert_++;
            search_pool_.insert(cur_neighbor, tmp_dist);
            // is_in_head |= search_pool_.insert(cur_neighbor, tmp_dist);

            // memory::mem_prefetch_l2(
            //     reinterpret_cast<const char*>(get_vector(search_pool_.next_id())), 10
            // );
        }
#if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        collector_insert_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
#endif
    }

    update_results(res_pool, query);
    res_pool.copy_results(results);
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    // timespec u1, u2;
    // int64_t cpu_time = 0;
    // uint64_t wall_time;
    scan_count_++;

    // auto t1 = std::chrono::high_resolution_clock::now();
    // clock_gettime(RUSAGE_SELF, &u1);

    // wait for (dimension_ << 3) nanoseconds
    // for (volatile size_t i = 0; i < (degree_bound_ << 2); i++) { }

    volatile float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);
    // float sqr_y = 0;
    // float sqr_y = l2_c_;

    /* Compute approximate distance by Fast Scan */
    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist,
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        packed_code,
        factor
    );
    // auto t2 = std::chrono::high_resolution_clock::now();
    // wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    // cpu_time = (static_cast<int64_t>(u2.tv_sec) - static_cast<int64_t>(u1.tv_sec)) * 1e9 +
    //            (static_cast<int64_t>(u2.tv_nsec) - static_cast<int64_t>(u1.tv_nsec));
    // ` += wall_time;
    // scanner_cpu_time_ += cpu_time;

    // t1 = std::chrono::high_resolution_clock::now();
    // clock_gettime(RUSAGE_SELF, &u1);
    uint64_t is_in_head = 0;
    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        is_in_head |= search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }
    num_is_in_head_ += is_in_head;
    // clock_gettime(RUSAGE_SELF, &u2);
    // t2 = std::chrono::high_resolution_clock::now();
    // wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    // cpu_time = (static_cast<int64_t>(u2.tv_sec) - static_cast<int64_t>(u1.tv_sec)) * 1e9 +
    //            (static_cast<int64_t>(u2.tv_nsec) - static_cast<int64_t>(u1.tv_nsec));
    // collector_wall_time_ += wall_time;
    // collector_cpu_time_ += (cpu_time - collector_cpu_time_) / collector_count_;

    return sqr_y;
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, space::l2_sqr(query, get_vector(cur_neighbor), dimension_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

inline void QuantizedGraph::initialize() {
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    this->code_offset_ = dimension_;  // Pos of packed code (aligned)
    this->factor_offset_ =
        code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
    this->neighbor_offset_ =
        factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
    this->row_offset_ = neighbor_offset_ + degree_bound_;

    /* Allocate memory of data*/
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
}

// find candidate neighbors for cur_id, exclude the vertex itself
inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        auto sqr_y = scan_neighbors(
            q_obj, get_vector(cur_candi), appro_dist.data(), tmp_pool, cur_degree
        );
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, sqr_y);
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
    RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
    x_pad.setZero();
    c_pad.setZero();

    /* Copy data */
    for (size_t i = 0; i < cur_degree; ++i) {
        auto neighbor_id = new_neighbors[i].id;
        const auto* cur_data = get_vector(neighbor_id);
        std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    const auto* cur_cent = get_vector(cur_id);
    std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

    /* rotate Matrix */
    RowMatrix<float> x_rotated(cur_degree, padded_dim_);
    RowMatrix<float> c_rotated(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
    }
    this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    // Get codes and factors for rabitq
    float* fac_ptr = get_factor(cur_id);
    float* triple_x = fac_ptr;
    float* factor_dq = triple_x + this->degree_bound_;
    float* factor_vq = factor_dq + this->degree_bound_;
    rabitq_codes(
        x_rotated, c_rotated, get_packed_code(cur_id), triple_x, factor_dq, factor_vq
    );
}
}  // namespace symqg
