

class ResBlock{
    Conv conv0;
    Conv conv1;
    Conv respool;
    bool pool;
    ResBlock(const int in_channel, const int out_channel, const int kernel_w, const int stride_w,
          const int pad_w, const int kernel_h = 0, const int stride_h = 0, const int pad_h = 0,
          const int pool_stride_w = 0, const int pool_stride_h = 0){
        if(pool_stride_w == 0 && pool_stride_h == 0){
            pool = 0;
            conv0 = Conv(in_channel, out_channel, kernel_w, stride_w, pad_w, kernel_h, stride_h, pad_h);
        }else{
            pool = 1;
            conv0 = Conv(in_channel, out_channel, kernel_w, pool_stride_w, pad_w, kernel_h, pool_stride_h, pad_h);
            respool = Conv(in_channel, out_channel, 1, pool_stride_w, 0, 1, pool_stride_h, 0);
        }
        conv1 = Conv(out_channel, out_channel, kernel_w, stride_w, pad_w, kernel_h, stride_h, pad_h);
    }

    float* forward(TensorDiscriptor *input){
        TensorDiscriptor *feat = conv0(input);
        //feat = Relu(feat);
        feat = conv1(feat);
        TensorDiscriptor *residual;
        if(pool) residual = respool(input);
        else residual = input;
        //feat = add(feat, residual);
        return feat->tensor;
    }
};

class ResNet18{

    Conv conv0;
    MaxPool mpool;
    GlobalAvgPool apool;
    ResBlock blocks[8];
    Gemm gemm;
    bool ResPool = {0,0,1,0,1,0,1,0};
    ResNet18(){
        conv0 = Conv(3, 64, 7, 2, 3);
        mpool = MaxPool(3, 2, 1);
        int cur_ch = 64;
        for(int i = 0;i < 8;++i){
            if(ResPool[i]){
                blocks[i] = ResBlock(cur_ch, cur_ch * 2, 3, 1, 1, pool_stride_w = 2, pool_stride_h = 2);
                cur_ch *= 2;
            }else{
                blocks[i] = ResBlock(cur_ch, cur_ch, 3, 1, 1);
            }
        }
        apool = GlobalAvgPool();
        gemm = Gemm(512, 1000);
    }

    float* forward(TensorDiscriptor *input){
        TensorDiscriptor *feature = conv0(input);
        feature = mpool(feature);
        for(int i = 0;i < 8;++i){
            feature = blocks[i](feature);
        }
        feature = apool(feature);
        //flatten
        feature = gemm(feature);
        return feature->tensor;
    }
};
