#pragma once

#include <cstddef>
#include <vector>

#include "../common.hpp"
#include "./memory.hpp"

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
    size_t w_collector_, w_scanner_, num_collector_, num_scanner_;
    size_t h_;          /* height of each bucket */
    size_t capacity_;   /* capacity == w_collector_ * num_collector_ * h_ */
    std::vector<SearchBuffer> buckets_;
    std::vector<PID> strip_;

    static void set_checked(PID& data_id) { data_id |= (1 << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

   public:
    BucketBuffer() = default;

    explicit BucketBuffer(size_t w_collector, size_t w_scanner, size_t num_collector, size_t num_scanner,
                 size_t h)
        : w_collector_(w_collector),
          w_scanner_(w_scanner),
          num_collector_(num_collector),
          num_scanner_(num_scanner), 
          h_(h),
          capacity_(w_collector_ * num_collector_ * h_)
        {

            this->buckets_ = std::vector<SearchBuffer>(this->w_collector_ * this->num_collector_);
            this->strip_ = std::vector<PID>(this->w_scanner_ * this->num_scanner_);

            for (size_t i = 0; i < this->w_collector_ * this->num_collector_; ++i) {
                this->buckets_[i] = SearchBuffer(this->h_);
            }
        }
    
    [[nodiscard]] auto is_full(size_t collector_id, float dist) const -> bool {
        for (size_t i = this->w_collector_ * collector_id; i < this->w_collector_ * (collector_id + 1); ++i) {
            if (!this->buckets_[i].is_full(dist)) {
                return false;
            }
        }
        return true;
    }

    /* shrink or expand h_ by a factor of new_size/capacity_ */
    void resize(size_t new_size) {
        size_t factor;

        if (new_size > this->capacity_) {
            factor = new_size / this->capacity_;
            this->h_ *= factor;
            this->capacity_ = this->w_collector_ * this->num_collector_ * this->h_;
        } else {
            factor = this->capacity_ / new_size;
            this->h_ /= factor;
            this->capacity_ = this->w_collector_ * this->num_collector_ * this->h_;

            assert(this->h_ >= this->h_);
        }

        for (size_t i = 0; i < this->w_collector_ * this->num_collector_; ++i) {
            this->buckets_[i].resize(this->h_);
        }
    }
    
    PID pop(size_t scanner_id) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        while (true) {
            for (size_t i = strip_start; i < strip_end; ++i) {
                if (!is_checked(this->strip_[i])) {
                    return this->strip_[i];
                }
            }
        }
    }

    void insert(size_t collector_id, PID data_id, float dist) {
        /* select a random bucket not full from collector's buckets */
        size_t bucket_id = rand() % this->w_collector_ + collector_id * this->w_collector_;
        for (size_t i = bucket_id; i < this->w_collector_ * (collector_id + 1); ++i) {
            if (!this->buckets_[i].is_full(dist)) {
                bucket_id = i;
                break;
            }
            if (i == this->w_collector_ * (collector_id + 1) - 1) {
                i = this->w_collector_ * collector_id;
            }
        }

        SearchBuffer& bucket = this->buckets_[bucket_id];
        size_t strip_start = collector_id * this->w_collector_;
        size_t strip_end = strip_start + this->w_collector_;

        if (dist <= bucket.next_dist()) {
            /* no need to insert into bucket, just insert into strip */
            while (true) {
                for (size_t i = strip_start; i < strip_end; ++i) {
                    if (is_checked(this->strip_[i])) {
                        this->strip_[i] = data_id;
                        return;
                    }
                }
            }
        } else {
            PID pid = bucket.pop();
            /* insert into bucket */
            bucket.insert(data_id, dist);

            /* insert into strip */
            while (true) {
                for (size_t i = strip_start; i < strip_end; ++i) {
                    if (is_checked(this->strip_[i])) {
                        this->strip_[i] = pid;
                        return;
                    }
                }
            }
        }
    }

    void clear() {
        for (size_t i = 0; i < this->w_collector_ * this->num_collector_; ++i) {
            this->buckets_[i].clear();
        }
    }
    
    
};

class ApproDistBatch {
   private:
    float* dist;        /* approximate distance of pid's neighbors */
    size_t degree_;     /* degree of pid */
    PID pid;
    uint32_t valid;

   public:
    ApproDistBatch() = default;

    explicit ApproDistBatch(size_t degree, PID pid) : degree_(degree), pid(pid), dist(new float[degree]), valid(0) {}

    explicit ApproDistBatch(size_t degree) : degree_(degree), pid(0), dist(new float[degree]), valid(0) {}

    ~ApproDistBatch() { delete[] dist; }

    void clear() {
        this->degree_ = 0;
        this->pid = 0;
        delete[] dist;
    }
};

class Strip {
   private:
    float* dist_;
    size_t* pids_;
    size_t w_scanner_, num_scanner_;
    size_t w_collector_, num_collector_;
    size_t w_;          /* mininal operation width ( == max_degree) */
    size_t capacity_;   /* capacity == w_scanner_ * num_scanner_ * w_ == w_collector_ * num_collector_ * w_ */
    
    enum StripState {
        SCANNING,
        SCANNED,
        COLLECTING,
        COLLECTED
    };
    StripState* state_;

   public:
    Strip() = default;

    explicit Strip(size_t w_, size_t w_scanner, size_t num_scanner, size_t w_collector, size_t num_collector)
        : w_scanner_(w_scanner), 
        num_scanner_(num_scanner), 
        w_collector_(w_collector), 
        num_collector_(num_collector), 
        w_(w_), 
        capacity_(w_scanner * num_scanner * w_),
        dist_(new float[capacity_]),
        pids_(new size_t[w_scanner * num_scanner]),
        state_(new StripState[w_scanner * num_scanner])
        {
            assert(this->w_scanner_ * this->num_scanner_ == this->w_collector_ * this->num_collector_);
        }
    
    void set_scanned(size_t& scanner_id, PID& pid) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        for (size_t i = strip_start; i < strip_end; ++i) {
            if (this->pids_[i] == pid) {
                assert(this->state_[i] == StripState::SCANNING);
                this->state_[i] = StripState::SCANNED;
                return;
            }
        }

        throw std::runtime_error("pid not found in scanner's strip");
    }

    void set_collected(size_t& collector_id, PID& pid) {
        size_t strip_start = collector_id * this->w_collector_;
        size_t strip_end = strip_start + this->w_collector_;

        for (size_t i = strip_start; i < strip_end; ++i) {
            if (this->pids_[i] == pid) {
                assert(this->state_[i] == StripState::COLLECTING);
                this->state_[i] = StripState::COLLECTED;
                return;
            }
        }

        throw std::runtime_error("pid not found in collector's strip");
    }

    [[nodiscard]] float* get_scanner_buffer(size_t scanner_id, PID pid) {
        size_t strip_start = scanner_id * this->w_scanner_;
        size_t strip_end = strip_start + this->w_scanner_;

        while (true) {
            for (size_t i = strip_start; i < strip_end; ++i) {
                if (this->state_[i] == StripState::COLLECTED) {
                    this->state_[i] = StripState::SCANNING;
                    return this->dist_ + i * this->w_;
                }
            }
        }
    }

    [[nodiscard]] std::pair<size_t, float*> get_collector_strip(size_t collector_id) {
        size_t strip_start = collector_id * this->w_collector_;
        size_t strip_end = strip_start + this->w_collector_;

        while (true) {
            for (size_t i = strip_start; i < strip_end; ++i) {
                if (this->state_[i] == StripState::COLLECTED) {
                    this->state_[i] = StripState::SCANNING;
                    return std::make_pair(this->pids_[i], this->dist_ + i * this->w_);
                }
            }
        }
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