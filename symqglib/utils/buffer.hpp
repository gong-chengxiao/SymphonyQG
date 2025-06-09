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
    size_t insert(PID data_id, float dist) {
        size_t lo = binary_search(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Candidate<float>));
        data_[lo] = Candidate<float>(data_id, dist);
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
        return static_cast<size_t>(cur_ == lo);
    }

    size_t insert(Candidate<float> candidate) {
        size_t lo = binary_search(candidate.distance);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Candidate<float>));
        data_[lo] = candidate;
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
        return static_cast<size_t>(cur_ == lo);
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
            this->buffer_ = std::vector<PID>(this->h_buffer_, NOT_FOUND);

            if ((this->h_buffer_ & (this->h_buffer_ - 1)) != 0) {
                throw std::runtime_error("h_buffer_ is not power of 2");
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
        return this->buffer_has_next() || this->bucket_.has_next();
    }

    void resize(size_t new_size) {
        this->h_bucket_ = new_size;
        this->bucket_.resize(this->h_bucket_);
    }

    void resize_buffer(size_t new_size) {
        this->buffer_ = std::vector<PID>(new_size, NOT_FOUND);
        this->h_buffer_ = new_size;
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

    [[nodiscard]] auto pop_from_bucket() -> PID {
        return this->bucket_.pop();
    }

    [[nodiscard]] auto next_id_from_bucket() -> PID {
        return this->bucket_.next_id();
    }

    [[nodiscard]] auto bucket_has_next() const -> bool {
        return this->bucket_.has_next();
    }

    size_t insert(PID data_id, float dist) {
        return this->bucket_.insert(data_id, dist);
    }

    size_t insert(Candidate<float> candidate) {
        return this->bucket_.insert(candidate);
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
    size_t w_ = 8;          /* align candidates to cache line (cacheline=64B, size(Candidate<float>)=8B) */
    size_t length_;
    size_t collector_pos_;  /* collector position in `pids_`. all pids before collector_pos_
                               are collected */
    size_t scanner_pos_;    /* scanner position in `pids_` */
    std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>> candidates_;

   public:
    Strip() = default;

    explicit Strip(size_t length)
        : length_(length), collector_pos_(0), scanner_pos_(w_), candidates_(length) {
            if ((length_ & (w_ - 1)) != 0) {
                std::cerr << "length_ is not multiple of w_" << std::endl;
                throw std::runtime_error("length_ is not multiple of w_");
            }
            if (length_ < 2) {
                std::cerr << "length_ must be greater than 1" << std::endl;
                throw std::runtime_error("length_ must be greater than 1");
            }
            std::fill(candidates_.begin(), candidates_.end(), Candidate<float>(0, 0));
        }

    void clear() {
        this->collector_pos_ = 0;
        this->scanner_pos_ = w_;
        std::fill(candidates_.begin(), candidates_.end(), Candidate<float>(0, 0));
    }

    void resize(size_t new_length) {
        if (new_length < 2) {
            std::cerr << "length_ must be greater than 1" << std::endl;
            throw std::runtime_error("length_ must be greater than 1");
        }
        if ((new_length & (w_ - 1)) != 0) {
            std::cerr << "length_ is not multiple of w_" << std::endl;
            throw std::runtime_error("length_ is not multiple of w_");
        }

        this->length_ = new_length;
        this->candidates_ = std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(new_length, Candidate<float>(0, 0));
    }

    /* put a candidate into strip, return false if strip may be full in next round */
    bool put(PID pid, float dist) {
        candidates_[scanner_pos_ & (length_ - 1)] = Candidate<float>(pid, dist);
        bool stalled = ((scanner_pos_ + 1) & ~(w_ - 1)) == (collector_pos_ & ~(w_ - 1));
        scanner_pos_ += static_cast<size_t>(!stalled);
        return !stalled;
    }

    /* retrieve a candidate from strip, return false if strip may be empty in next round */
    [[nodiscard]] auto retrieve() -> Candidate<float> {
        bool stalled = ((collector_pos_ + 1) & ~(w_ - 1)) == (scanner_pos_ & ~(w_ - 1));
        collector_pos_ += static_cast<size_t>(!stalled);
        return stalled ? Candidate<float>(0, 0) : candidates_[collector_pos_ & (length_ - 1)];
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