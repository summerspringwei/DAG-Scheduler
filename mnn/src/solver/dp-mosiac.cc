float latency(float* accArr, int i, int j){

}

void dp_solve(float *arr, const int length){
    float result = 100000;
    for(int i=0; i<length; ++i){
        for(int j=i+1;j<length; ++j){
            float tmp_result = latency(arr, 0, i) + latency(arr, i+1, j);
        }
    }
}