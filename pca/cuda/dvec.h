#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <cuda_runtime.h>

template <class T>
class dvec{
private:
    T* start_;
    T* end_;


public:
    explicit dvec()
        : start_(0),
          end_(0){}

    explicit dvec(size_t size){
       cudaMalloc((void**)&start_, size * sizeof(T));
       end_ = start_ + size;
    }
    
    ~dvec(){
        if (start_ != 0){
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    size_t size() const{
        return end_ - start_;
    }

    const T* data() const{
        return start_;
    }

    T* data(){
        return start_;
    }

    void set(const T* src){
        cudaMemcpy(start_, src, size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void get(T* dest){
        cudaMemcpy(dest, start_, size() * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

#endif