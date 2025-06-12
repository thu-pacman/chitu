/**
 * This file has adaption of open-source code from the following sources:
 * - https://github.com/kvcache-ai/ktransformers, licensed under Apache 2.0.
 */

#include "shared_mem_buffer.h"
#include <cstdio>

SharedMemBuffer::SharedMemBuffer() {
    buffer_ = nullptr;
    size_ = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
    if (buffer_) {
        free(buffer_);
    }
}

void SharedMemBuffer::alloc(
    void *object, std::vector<std::pair<void **, uint64_t>> requests) {
    uint64_t size = 0;
    for (auto &request : requests) {
        size += request.second;
    }
    if (size > size_) {
        if (buffer_) {
            free(buffer_);
        }
        buffer_ = malloc(size);
        size_ = size;
        for (auto &obj_requests : hist_requests_) {
            for (auto &requests : obj_requests.second) {
                arrange(requests);
            }
        }
    }
    arrange(requests);
    hist_requests_[object].push_back(requests);
}

void SharedMemBuffer::dealloc(void *object) { hist_requests_.erase(object); }

void SharedMemBuffer::arrange(
    std::vector<std::pair<void **, uint64_t>> requests) {
    uint64_t offset = 0;
    for (auto &request : requests) {
        *(request.first) = (uint8_t *)buffer_ + offset;
        offset += request.second;
    }
}
