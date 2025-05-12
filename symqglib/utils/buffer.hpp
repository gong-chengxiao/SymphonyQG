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
    
    [[nodiscard]] auto is_full(float dist) const -> bool {
        return this->bucket_.is_full(dist);
    }

    [[nodiscard]] auto has_next() const -> bool {
        return this->bucket_.has_next();
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
        if (!this->bucket_.is_full(dist)) {
            this->bucket_.insert(data_id, dist);
        }
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
   public:
    enum StripState {
        COLLECTED,
        SCANNING,
        SCANNED,
        COLLECTING
    };

   private:
    float* dist_;
    size_t w_;          /* mininal operation width ( == max_degree) */
    size_t size_;
    size_t length_;
    size_t collector_pos_;
    size_t scanner_pos_;

    struct alignas(64) ArrayData {
        std::array<std::atomic<StripState>, 2> state_;
        std::array<PID, 2> pids_;
    } data_;

   public:
    Strip() = default;

    explicit Strip(size_t w_)
        : w_(w_), 
        size_(w_ << 1),
        length_(2),
        collector_pos_(0),
        scanner_pos_(0) {
            dist_ = new float[size_];
            for (size_t i = 0; i < length_; ++i) {
                this->data_.state_[i] = StripState::COLLECTED;
            }
        }
    
    ~Strip() {
        delete[] dist_;
    }

    void clear() {
        for (size_t i = 0; i < length_; ++i) {
            this->data_.state_[i] = StripState::COLLECTED;
        }
    }

    /* return if any strip is collecting */
    [[nodiscard]] bool is_collecting() const {
        return this->data_.state_[collector_pos_] == StripState::COLLECTING;
    }

    /* return if any strip is scanned */
    [[nodiscard]] bool is_scanned() const {
        return this->data_.state_[scanner_pos_] == StripState::SCANNED;
    }

    void set_scanned() {
        StripState expected = StripState::SCANNING;
        const StripState desired = StripState::SCANNED;
        
        if (this->data_.state_[scanner_pos_].compare_exchange_strong(expected, desired)) {
            return;
        }

        throw std::runtime_error("no SCANNING state found");
    }

    void set_collected() {
        StripState expected = StripState::COLLECTING;
        const StripState desired = StripState::COLLECTED;

        if (this->data_.state_[collector_pos_].compare_exchange_strong(expected, desired)) {
            return;
        }

        throw std::runtime_error("no COLLECTING state found");
    }

    /* spin until get a buffer with COLLECTED state */
    [[nodiscard]] float* try_get_scanner_buffer(PID pid) {
        size_t new_scanner_pos = scanner_pos_ ^ 1;

            StripState expected = StripState::COLLECTED;
            const StripState desired = StripState::SCANNING;

            if (this->data_.state_[new_scanner_pos].compare_exchange_strong(expected, desired)) {
                this->data_.pids_[new_scanner_pos] = pid;
                this->scanner_pos_ = new_scanner_pos;
                return this->dist_ + (new_scanner_pos ? this->w_ : 0);
            }

        return nullptr;
    }

    /* try getting a buffer with SCANNED state, return std::pari(1U << 31, nullptr) if fail */
    [[nodiscard]] std::pair<PID, float*> try_get_collector_buffer() {
        size_t new_collector_pos = collector_pos_ ^ 1;

        StripState expected = StripState::SCANNED;
        const StripState desired = StripState::COLLECTING;

        if (this->data_.state_[new_collector_pos].compare_exchange_strong(expected, desired)) {
            this->collector_pos_ = new_collector_pos;
            return std::make_pair(this->data_.pids_[new_collector_pos], this->dist_ + (new_collector_pos ? this->w_ : 0));
        }

        return std::make_pair(NOT_FOUND, nullptr);
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