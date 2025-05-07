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
    size_t w_collector_, w_scanner_, num_collectors_, num_scanners_;    /* w_collector_ * num_collectors_ == w_scanner_ * num_scanners_ */
    size_t h_bucket_;           /* height of each bucket */
    size_t h_strip_;            /* height of strip */
    size_t capacity_;           /* total capacity of buckets == w_collector_ * num_collectors_ * h_bucket_ */
    std::vector<SearchBuffer> buckets_;
    std::vector<PID> strip_;   /* size == w_scanner_ * num_scanners_ * h_strip_ */

    static void set_checked(PID& data_id) { data_id |= (1U << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

   public:
    BucketBuffer() = default;

    explicit BucketBuffer(size_t w_collector, size_t w_scanner, size_t num_collectors, size_t num_scanners,
                 size_t h_bucket, size_t h_strip)
        : w_collector_(w_collector),
          w_scanner_(w_scanner),
          num_collectors_(num_collectors),
          num_scanners_(num_scanners), 
          h_bucket_(h_bucket),
          h_strip_(h_strip),
          capacity_(w_collector_ * num_collectors * h_bucket_)
        {
            assert(w_collector_ * num_collectors == w_scanner_ * num_scanners_);

            this->buckets_ = std::vector<SearchBuffer>(this->w_collector_ * this->num_collectors_);
            this->strip_ = std::vector<PID>(this->w_scanner_ * this->num_scanners_ * this->h_strip_);

            for (size_t i = 0; i < this->w_collector_ * this->num_collectors_; ++i) {
                this->buckets_[i] = SearchBuffer(this->h_bucket_);
            }
            for (size_t i = 0; i < this->w_scanner_ * this->num_scanners_ * this->h_strip_; ++i) {
                this->strip_[i] = NOT_FOUND;
            }
        }

    void clear() {
        for (size_t i = 0; i < this->w_collector_ * this->num_collectors_; ++i) {
            this->buckets_[i].clear();
        }
        for (size_t i = 0; i < this->w_scanner_ * this->num_scanners_ * this->h_strip_; ++i) {
            this->strip_[i] = NOT_FOUND;
        }
    }
    
    [[nodiscard]] auto is_full(size_t collector_id, float dist) const -> bool {
        size_t bucket_start = this->w_collector_ * collector_id;
        size_t bucket_end = bucket_start + this->w_collector_;

        for (size_t i = bucket_start; i < bucket_end; ++i) {
            if (!this->buckets_[i].is_full(dist)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] auto has_next() const -> bool {
        for (size_t i = 0; i < num_collectors_ * w_collector_; ++i) {
            if (this->buckets_[i].has_next()) {
                return true;
            }
        }
        return false;
    }

    /* shrink or expand h_bucket_ by a factor of new_size/capacity_ */
    void resize(size_t new_size) {
        size_t factor;

        if (this->capacity_ == 0) {
            this->capacity_ = new_size;
            this->h_bucket_ = new_size / (this->w_collector_ * this->num_collectors_);
        } else if (new_size > this->capacity_) {
            factor = new_size / this->capacity_;
            this->h_bucket_ *= factor;
            this->capacity_ = this->w_collector_ * this->num_collectors_ * this->h_bucket_;
        } else {
            factor = this->capacity_ / new_size;
            this->h_bucket_ /= factor;
            this->capacity_ = this->w_collector_ * this->num_collectors_ * this->h_bucket_;
        }

        for (size_t i = 0; i < this->w_collector_ * this->num_collectors_; ++i) {
            this->buckets_[i].resize(this->h_bucket_ + 1);
        }
    }
    
    PID try_pop(size_t scanner_id) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        for (size_t i = strip_start * this->h_strip_; i < strip_end * this->h_strip_; ++i) {
            if (!is_checked(this->strip_[i])) {
                PID pid = this->strip_[i];
                set_checked(this->strip_[i]);
                return pid;
            }
        }
        return NOT_FOUND;
    }

    void insert(size_t collector_id, PID data_id, float dist) {
        size_t start = this->w_collector_ * collector_id;
        size_t end = start + this->w_collector_;
        
        for (size_t i = start; i < end; ++i) {
            if (!this->buckets_[i].is_full(dist)) {
                this->buckets_[i].insert(data_id, dist);
                return;
            }
        }
    }

    void try_promote(size_t collector_id) {
        size_t start = this->w_collector_ * collector_id;
        size_t end = start + this->w_collector_;

        for (size_t i = start; i < end; ++i) {
            for (size_t j = i * this->h_strip_; j < (i + 1) * this->h_strip_; ++j) {
                if (is_checked(this->strip_[j]) && this->buckets_[i].has_next()) {
                    this->strip_[j] = this->buckets_[i].pop();
                }
            }
        }
    }
    
};

class Strip {
   private:
    float* dist_;
    std::vector<PID> pids_;
    size_t w_scanner_, num_scanners_;
    size_t w_collector_, num_collectors_;
    size_t w_;          /* mininal operation width ( == max_degree) */
    size_t capacity_;   /* capacity == w_scanner_ * num_scanners_ * w_ == w_collector_ * num_collectors_ * w_ */
    
    enum StripState {
        COLLECTED,
        SCANNING,
        SCANNED,
        COLLECTING
    };
    std::vector<std::atomic<StripState>> state_;
    std::atomic<size_t> num_collecting_, num_scanned_;

   public:
    Strip() = default;

    explicit Strip(size_t w_, size_t w_scanner, size_t num_scanners, size_t w_collector, size_t num_collectors)
        : w_scanner_(w_scanner), 
        num_scanners_(num_scanners), 
        w_collector_(w_collector), 
        num_collectors_(num_collectors), 
        w_(w_), 
        capacity_(w_scanner * num_scanners * w_),
        state_(w_scanner * num_scanners),
        num_collecting_(0),
        num_scanned_(0)
        {
            assert(this->w_scanner_ * this->num_scanners_ == this->w_collector_ * this->num_collectors_);

            dist_ = new float[capacity_];
            pids_ = std::vector<PID>(w_scanner_ * num_scanners_);
            for (size_t i = 0; i < this->w_scanner_ * this->num_scanners_; ++i) {
                this->state_[i] = StripState::COLLECTED;
            }
        }
    
    ~Strip() {
        delete[] dist_;
    }

    void clear() {
        for (size_t i = 0; i < this->w_scanner_ * this->num_scanners_; ++i) {
            this->state_[i] = StripState::COLLECTED;
        }
        this->num_collecting_.store(0);
        this->num_scanned_.store(0);
    }

    /* return if any strip is collecting */
    [[nodiscard]] bool is_collecting() const {
        return this->num_collecting_.load() > 0;
    }

    /* return if any strip is scanned */
    [[nodiscard]] bool is_scanned() const {
        return this->num_scanned_.load() > 0;
    }

    void set_scanned(size_t scanner_id) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        for (size_t i = strip_start; i < strip_end; ++i) {
            StripState expected = StripState::SCANNING;
            StripState desired = StripState::SCANNED;
            
            if (this->state_[i].compare_exchange_strong(expected, desired)) {
                this->num_scanned_.fetch_add(1);
                return;
            }
        }

        throw std::runtime_error("no SCANNING state found in scanner's strip");
    }

    void set_collected(size_t collector_id) {
        size_t strip_start = collector_id * this->w_collector_;
        size_t strip_end = strip_start + this->w_collector_;

        for (size_t i = strip_start; i < strip_end; ++i) {
            StripState expected = StripState::COLLECTING;
            StripState desired = StripState::COLLECTED;

            if (this->state_[i].compare_exchange_strong(expected, desired)) {
                this->num_collecting_.fetch_sub(1);
                return;
            }
        }

        throw std::runtime_error("no COLLECTING state found in collector's strip");
    }

    /* endless loop until get a buffer with COLLECTED state */
    [[nodiscard]] float* get_scanner_buffer(size_t scanner_id, PID pid) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        while (true) {
            for (size_t i = strip_start; i < strip_end; ++i) {
                StripState expected = StripState::COLLECTED;
                StripState desired = StripState::SCANNING;

                if (this->state_[i].compare_exchange_strong(expected, desired)) {
                    this->pids_[i] = pid;
                    return this->dist_ + i * this->w_;
                }
            }
        }
    }

    /* try getting a buffer with SCANNED state, return std::pari(1U << 31, nullptr) if fail */
    [[nodiscard]] std::pair<PID, float*> try_get_collector_buffer(size_t collector_id) {
        size_t strip_start = collector_id * this->w_collector_;
        size_t strip_end = strip_start + this->w_collector_;

        for (size_t i = strip_start; i < strip_end; ++i) {
            StripState expected = StripState::SCANNED;
            StripState desired = StripState::COLLECTING;

            if (this->state_[i].compare_exchange_strong(expected, desired)) {
                this->num_collecting_.fetch_add(1);
                this->num_scanned_.fetch_sub(1);
                return std::make_pair(this->pids_[i], this->dist_ + i * this->w_);
            }
        }
        return std::make_pair(1U << 31, nullptr);
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