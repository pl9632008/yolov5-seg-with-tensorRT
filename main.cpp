#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

float data[3*640*640];
float prob0[25200*117];
float prob1[32*160*160];

struct  Object{
  Rect_<float> rect;
  int label;
  float prob;
  vector<float>maskdata;
  Mat mask;
};

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


//先对原图进行padding,宽和高哪边小，就对哪边进行padding，然后缩放到输入网络的图片宽高。
Mat preprocess_img(cv::Mat& img, int input_w, int input_h,int & padw,int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows))); 
    padw = (input_w - w)/2;
    padh = (input_h - h)/2;
    return out;
}

void qsort_descent_inplace(vector<Object>&faceobjects,int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while (i<=j){
        while (faceobjects[i].prob>p ){
            i++;
        }
        while (faceobjects[j].prob<p){
            j--;
        }
        if(i<=j){
            swap(faceobjects[i],faceobjects[j]);
            i++;
            j--;
        }

    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void  qsort_descent_inplace(vector<Object>&faceobjects){
    if(faceobjects.empty()){
        return ;
    }
    qsort_descent_inplace(faceobjects,0,faceobjects.size()-1);
}

float intersection_area(Object & a,Object&b) {
    Rect2f inter = a.rect&b.rect;
    return inter.area();

}

void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
         Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
          Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
          "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush"
    };
    
    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };
    
    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }
    imwrite("./testout.jpg",image);
}


int main(int argc,char ** argv){

        size_t size{0};
        char * trtModelStream{nullptr};
        ifstream file("/wangjiadong/yolov5/yolov5x-seg.engine", ios::binary);
        
        if(file.good()){
            file.seekg(0,ios::end);
            size = file.tellg();
            file.seekg(0,ios::beg);
            trtModelStream = new char[size];
            file.read(trtModelStream,size);
            file.close();
        }
    
        IRuntime * runtime = createInferRuntime(logger);
        ICudaEngine * engine = runtime->deserializeCudaEngine(trtModelStream,size);
        IExecutionContext *context = engine->createExecutionContext();
        delete[] trtModelStream;

        int BATCH_SIZE=1;
        int INPUT_H=640;
        int INPUT_W=640;

        const char * images = "images";
        const char * output0 = "output0";
        const char * output1 = "output1";

        int32_t images_index = engine->getBindingIndex(images);
        int32_t output0_index = engine->getBindingIndex(output0);
        int32_t output1_index = engine->getBindingIndex(output1);

        cout<<images_index<<" "
            <<output0_index<<" "
            <<output1_index<<" "
            <<endl;
        cout<<engine->getNbBindings()<<endl;
      
        void * buffers[3];
        cudaMalloc(&buffers[images_index],BATCH_SIZE*3*INPUT_W*INPUT_H*sizeof(float));
        cudaMalloc(&buffers[output0_index],BATCH_SIZE*25200*117*sizeof(float));
        cudaMalloc(&buffers[output1_index],BATCH_SIZE*32*160*160*sizeof(float));

        Mat img = imread(argv[1]);

        int padw,padh;
        Mat pr_img = preprocess_img(img,INPUT_H,INPUT_W,padw,padh);

        // cv::imwrite("./test_primg.jpg",pr_img);

        for(int i = 0 ; i < INPUT_W*INPUT_H;i++){
            data[i] = pr_img.at<Vec3b>(i)[2]/255.0;
            data[i+INPUT_W*INPUT_H] = pr_img.at<Vec3b>(i)[1]/255.0;
            data[i+2*INPUT_W*INPUT_H]=pr_img.at<Vec3b>(i)[0]/255.0;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[images_index],data,BATCH_SIZE*3*INPUT_W*INPUT_H*sizeof(float),cudaMemcpyHostToDevice,stream);
        context->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(prob0,buffers[output0_index],1*25200*117*sizeof(float),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(prob1,buffers[output1_index],1*32*160*160*sizeof(float),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[images_index]);
        cudaFree(buffers[output0_index]);
        cudaFree(buffers[output1_index]);
        delete context;
        delete runtime;
        delete engine;

        vector<Object> objects;
       
        for(int i = 0 ; i<25200;++i){
            if(prob0[117*i+4]>0.5){
            
                int l,r,t,b;
                float r_w = INPUT_W/(img.cols*1.0);
                float r_h = INPUT_H/(img.rows*1.0);

                float x = prob0[117*i+0];
                float y = prob0[117*i+1];
                float w = prob0[117*i+2];
                float h = prob0[117*i+3];
                float score = prob0[117*i+4];

                vector<float> maskdata(prob0+117*(i+1)-32,prob0+117*(i+1));

                if(r_h>r_w){
                    l = x-w/2.0;
                    r = x+w/2.0;
                    t = y-h/2.0-(INPUT_H-r_w*img.rows)/2;
                    b = y+h/2.0-(INPUT_H-r_w*img.rows)/2;
                    l=l/r_w;
                    r=r/r_w;
                    t=t/r_w;
                    b=b/r_w;
                }else{
                    l = x-w/2.0-(INPUT_W-r_h*img.cols)/2;
                    r = x+w/2.0-(INPUT_W-r_h*img.cols)/2;
                    t = y-h/2.0;
                    b = y+h/2.0;
                    l=l/r_h;
                    r=r/r_h;
                    t=t/r_h;
                    b=b/r_h;
                }
                int label_index = max_element(prob0+117*i+5,prob0+117*(i+1)-32 ) - (prob0+117*i+5);
                Object obj;
                obj.rect.x = l;
                obj.rect.y = t;
                obj.rect.width=r-l;
                obj.rect.height=b-t;
                obj.label = label_index;
                obj.prob = score;
                obj.maskdata=maskdata;
                objects.push_back(obj);
                }

        }

        qsort_descent_inplace(objects);
        vector<int> picked;
        nms_sorted_bboxes(objects,picked,0.45);
        int count = picked.size();
        cout<<"count="<<count<<endl;
        vector<Object>obj_out(count);
        for(int i = 0 ; i <count ; ++i){
            obj_out[i] = objects[picked[i]];
        }

        for(int i = 0 ; i < count; i++){
            Object & obj = obj_out[i];
            Mat mask(160,160,CV_32FC1);
            mask = Scalar(0.f);
            for(int p = 0; p <32 ;p++){
                vector<float>temp(prob1+160*160*p,prob1+160*160*(p+1));
                float coeff = obj.maskdata[p];
                float *mp = (float *) mask.data;
                for(int j = 0 ; j<160*160;j++){
                    mp[j] += temp.data()[j]*coeff;
                }
            }
            float ratio = 640./160;//原始图到特征图的缩放比例,padding也要进行缩放
            Rect roi( int( padw/ratio) ,int( padh/ratio),  int((640-padw*2)/ratio), int((640-padh*2)/ratio));
            Mat dest;
            cv::exp(-mask,dest);
            dest = 1./(1.+dest);
            dest= dest(roi);
            Mat mask2;
            resize(dest,mask2,img.size());
            obj.mask= Mat(img.rows,img.cols,CV_8UC1);
            obj.mask = Scalar(0);
            for(int y = 0 ; y < img.rows; y++){
                if(y<obj.rect.y || y>obj.rect.y+obj.rect.height){
                    continue;
                }
                float*mp2 = mask2.ptr<float>(y);
                uchar * bmp = obj.mask.ptr<uchar>(y);
                for(int x = 0 ; x< img.cols; x++){
                    if(x < obj.rect.x || x>obj.rect.x + obj.rect.width){
                        continue;
                    }
                    bmp[x] = mp2[x]>0.5f? 255 : 0;
                    
                }
            }
        }
        draw_objects(img,obj_out);
  }

