#pragma once

#include <cstddef>
#include <vector>

#include "../common.hpp"
#include "./memory.hpp"
#include <iostream>

#define NOT_FOUND 1U << 31

namespace symqg::buffer {
// @brief sorted linear buffer, served as beam set
class SearchBuffer {
   private:
    std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>> data_;
    size_t size_ = 0, cur_ = 0, capacity_;

    [[nodiscard]] auto binary_search(float dist) const {
        size_t lo = 0;
        size_t len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += static_cast<size_t>(data_[lo + half - 1].distance < dist) * half;
        }
        return (lo < size_ && data_[lo].distance < dist) ? lo + 1 : lo;
    }

    static void set_checked(PID& data_id) { data_id |= (1 << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

   public:
    SearchBuffer() = default;

    explicit SearchBuffer(size_t capacity) : data_(capacity + 1), capacity_(capacity) {}

    void prefetch() {
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(data_.data()), 20
        );
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(data_.data() + 20), 20
        );
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(data_.data() + 40), 20
        );
    }

    // insert a data point into buffer
    void insert(PID data_id, float dist) {
        size_t lo = binary_search(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Candidate<float>));
        data_[lo] = Candidate<float>(data_id, dist);
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
    }

    [[nodiscard]] auto is_full(float dist) const -> bool {
        return size_ == capacity_ && dist > data_[size_ - 1].distance;
    }

    // get closest unchecked data point
    PID pop() {
        PID cur_id = data_[cur_].id;
        set_checked(data_[cur_].id);
        ++cur_;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
            ++cur_;
        }
        return cur_id;
    }

    void clear() {
        size_ = 0;
        cur_ = 0;
    }

    [[nodiscard]] auto next_id() const { return data_[cur_].id; }

    [[nodiscard]] auto next_dist() const { return data_[cur_].distance; }

    [[nodiscard]] auto has_next() const -> bool { return cur_ < size_; }

    [[nodiscard]] auto size() const { return size_; }

    void resize(size_t new_size) {
        this->capacity_ = new_size;
        data_ = std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(
            capacity_ + 1
        );
    }
};

class BucketBuffer {
   private:
    size_t h_bucket_;   /* height of bucket */
    size_t h_buffer_;   /* height of buffer */
    SearchBuffer bucket_;
    std::vector<PID> buffer_;

    static void set_checked(PID& data_id) { data_id |= (1U << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

    [[nodiscard]] bool buffer_has_next() const {
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            if (!is_checked(this->buffer_[i])) {
                return true;
            }
        }
        return false;
    }

   public:
    BucketBuffer() = default;

    explicit BucketBuffer(size_t h_bucket, size_t h_buffer)
        : h_bucket_(h_bucket),
          h_buffer_(h_buffer)
        {
            this->bucket_ = SearchBuffer(this->h_bucket_);
            this->buffer_ = std::vector<PID>(this->h_buffer_);

            for (size_t i = 0; i < this->h_buffer_; ++i) {
                this->buffer_[i] = NOT_FOUND;
            }
        }

    void clear() {
        this->bucket_.clear();
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            this->buffer_[i] = NOT_FOUND;
        }
    }

    void prefetch_bucket() { bucket_.prefetch(); }

    [[nodiscard]] auto is_full(float dist) const -> bool {
        return this->bucket_.is_full(dist);
    }

    [[nodiscard]] auto has_next() const -> bool {
        /* order is important! */
        return this->buffer_has_next() || this->bucket_.has_next();
    }

    void resize(size_t new_size) {
        this->h_bucket_ = new_size;
        this->bucket_.resize(this->h_bucket_);
    }
    
    PID try_pop() {
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            if (!is_checked(this->buffer_[i])) {
                PID pid = this->buffer_[i];
                set_checked(this->buffer_[i]);
                return pid;
            }
        }
        return NOT_FOUND;
    }

    void insert(PID data_id, float dist) {
        this->bucket_.insert(data_id, dist);
    }

    void try_promote() {
        for (size_t i = 0; i < this->h_buffer_; ++i) {
            if (is_checked(this->buffer_[i]) && this->bucket_.has_next()) {
                this->buffer_[i] = this->bucket_.pop();
            }
        }
    }
    
};

class Strip {
   private:
    float* dist_;
    size_t w_;          /* mininal operation width ( == max_degree) */
    size_t length_;
    size_t size_;       /* size of dist_ == w_ * length_ */
    size_t collector_pos_;  /* collector position in `pids_`. all pids before collector_pos_ are collected */
    size_t scanner_pos_;    /* scanner position in `pids_` */
    std::vector<PID> pids_;

   public:
    Strip() = default;

    explicit Strip(size_t w_, size_t length)
        : w_(w_), 
        length_(length),
        size_(w_ * length_),
        collector_pos_(0),
        scanner_pos_(0),
        pids_(length) {
            dist_ = new float[size_];
        }
    
    ~Strip() {
        delete[] dist_;
    }

    void clear() {
        this->collector_pos_ = 0;
        this->scanner_pos_ = 0;
    }

    /* set scanned and return true if scanner may be stalled by collector in next round */
    bool set_scanned(PID pid) {
        pids_[scanner_pos_ & (length_ - 1)] = pid;
        bool stalled = ((scanner_pos_ + 1) & (length_ - 1)) == (collector_pos_ & (length_ - 1));
        scanner_pos_ += static_cast<size_t>(!stalled);
        return stalled;
    }

    void set_collected() {
        collector_pos_++;
    }

    [[nodiscard]] float* get_scanner() {
        return dist_ + (scanner_pos_ & (length_ - 1)) * w_;
    }

    [[nodiscard]] std::pair<PID, float*> try_get_collector() {
        if (collector_pos_ < scanner_pos_) {
            return std::make_pair(pids_[collector_pos_ & (length_ - 1)], dist_ + (collector_pos_ & (length_ - 1)) * w_);
        } else if (collector_pos_ == scanner_pos_) {
            return std::make_pair(NOT_FOUND, nullptr);
        }
        throw std::runtime_error("collector_pos_ > scanner_pos_");
    }
};

// sorted linear buffer to store search results
class ResultBuffer {
   public:
    explicit ResultBuffer(size_t capacity)
        : ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

    void insert(PID data_id, float dist) {
        if (size_ == capacity_ && dist > distances_[size_ - 1]) {
            return;
        }
        size_t lo = binary_search(dist);
        std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
        ids_[lo] = data_id;
        std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
        distances_[lo] = dist;
        size_ += static_cast<size_t>(size_ < capacity_);
    }

    [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

    const std::vector<PID, memory::AlignedAllocator<PID>>& ids() { return ids_; }

    void copy_results(PID* knn) const { std::copy(ids_.begin(), ids_.end() - 1, knn); }

   private:
    std::vector<PID, memory::AlignedAllocator<PID>> ids_;
    std::vector<float, memory::AlignedAllocator<float>> distances_;
    size_t size_ = 0, capacity_;

    [[nodiscard]] auto binary_search(float dist) const -> size_t {
        size_t lo = 0;
        size_t len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += static_cast<size_t>(distances_[lo + half - 1] < dist) * half;
        }
        return (lo < size_ && distances_[lo] < dist) ? lo + 1 : lo;
    }
};
}  // namespace symqg::buffer