// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer.h"
#include "net.h"

#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};


static ncnn::Net* yolo = 0;

static int detect_yolo(const unsigned char* rgba_data, int width, int height, std::vector<Object>& objects)
{
    if (!yolo)
    {
        yolo = new ncnn::Net;

        fprintf(stderr, "opt num_threads = %d\n", yolo->opt.num_threads);

        yolo->opt.use_vulkan_compute = true;
        // yolo->opt.use_bf16_storage = true;

//         yolo->opt.num_threads = 4;

//        yolo->register_custom_layer("yoloFocus", yoloFocus_layer_creator);

        // original pretrained model from https://github.com/ultralytics/yolo
        // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        yolo->load_param("model/yolo-fastest-opt.param");
        yolo->load_model("model/yolo-fastest-opt.bin");
    }

    const int target_size = 320;
//    const float prob_threshold = 0.25f;
//    const float nms_threshold = 0.45f;

    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba_data, ncnn::Mat::PIXEL_RGBA2RGB, width, height, w, h);

    // pad to target_size rectangle
    // yolo/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo->create_extractor();

    ex.input("data", in_pad);
    ncnn::Mat out;
    ex.extract("output", out);
    
    int count = out.h;
    
//    // apply nms with nms_threshold
//    std::vector<int> picked;
//    nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
//        objects[i] = proposals[picked[i]];
//
//        // adjust offset to original unpadded
//        float x0 = (objects[i].x - (wpad / 2)) / scale;
//        float y0 = (objects[i].y - (hpad / 2)) / scale;
//        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
//        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;
//
//        // clip
//        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
//
//        objects[i].x = x0;
//        objects[i].y = y0;
//        objects[i].w = x1 - x0;
//        objects[i].h = y1 - y0;
//    }
//
//    return 0;
//
//
//    for (int i = 0; i < out.h; i++)
//    {
        int label;
        float x1, y1, x2, y2, score;
        float pw,ph,cx,cy;
        const float* values = out.row(i);
        
        x1 = values[2] * width;
        y1 = values[3] * height;
        x2 = values[4] * width;
        y2 = values[5] * height;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;

        if(x1>width) x1=width;
        if(y1>height) y1=height;
        if(x2>width) x2=width;
        if(y2>height) y2=height;
        
        objects[i].label = label;
        objects[i].prob = score;
        objects[i].x = x1;
        objects[i].y = y1;
        objects[i].w = x2 - x1;
        objects[i].h = y2 - y1;
        
    }
    return 0;
}

    
//
//
//    std::vector<Object> proposals;
//
//    // anchor setting from yolo/models/yolos.yaml
//
//    // stride 8
//    {
//        ncnn::Mat out;
//        ex.extract("output", out);
//
//        ncnn::Mat anchors(6);
//        anchors[0] = 10.f;
//        anchors[1] = 13.f;
//        anchors[2] = 16.f;
//        anchors[3] = 30.f;
//        anchors[4] = 33.f;
//        anchors[5] = 23.f;
//
//        std::vector<Object> objects8;
//        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
//
//        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
//    }
//
//    // stride 16
//    {
//        ncnn::Mat out;
//        ex.extract("781", out);
//
//        ncnn::Mat anchors(6);
//        anchors[0] = 30.f;
//        anchors[1] = 61.f;
//        anchors[2] = 62.f;
//        anchors[3] = 45.f;
//        anchors[4] = 59.f;
//        anchors[5] = 119.f;
//
//        std::vector<Object> objects16;
//        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
//
//        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
//    }
//
//    // stride 32
//    {
//        ncnn::Mat out;
//        ex.extract("801", out);
//
//        ncnn::Mat anchors(6);
//        anchors[0] = 116.f;
//        anchors[1] = 90.f;
//        anchors[2] = 156.f;
//        anchors[3] = 198.f;
//        anchors[4] = 373.f;
//        anchors[5] = 326.f;
//
//        std::vector<Object> objects32;
//        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
//
//        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
//    }
//
//    // sort all proposals by score from highest to lowest
//    qsort_descent_inplace(proposals);
//
//    // apply nms with nms_threshold
//    std::vector<int> picked;
//    nms_sorted_bboxes(proposals, picked, nms_threshold);
//
//    int count = picked.size();
//
//    objects.resize(count);
//    for (int i = 0; i < count; i++)
//    {
//        objects[i] = proposals[picked[i]];
//
//        // adjust offset to original unpadded
//        float x0 = (objects[i].x - (wpad / 2)) / scale;
//        float y0 = (objects[i].y - (hpad / 2)) / scale;
//        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
//        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;
//
//        // clip
//        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
//
//        objects[i].x = x0;
//        objects[i].y = y0;
//        objects[i].w = x1 - x0;
//        objects[i].h = y1 - y0;
//    }
//
//    return 0;
//}

#include <emscripten.h>

static const unsigned char* rgba_data = 0;
static int w = 0;
static int h = 0;
static float* result_buffer = 0;

static ncnn::Mutex lock;
static ncnn::ConditionVariable condition;

static ncnn::Mutex finish_lock;
static ncnn::ConditionVariable finish_condition;

static void worker()
{
    while (1)
    {
        lock.lock();
        while (rgba_data == 0)
        {
            condition.wait(lock);
        }

        std::vector<Object> objects;
        detect_yolo(rgba_data, w, h, objects);

//         print_objects(objects);

        // result_buffer max 20 objects
        if (objects.size() > 20)
            objects.resize(20);

        size_t i = 0;
        for (; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            result_buffer[0] = obj.label;
            result_buffer[1] = obj.prob;
            result_buffer[2] = obj.x;
            result_buffer[3] = obj.y;
            result_buffer[4] = obj.w;
            result_buffer[5] = obj.h;

            result_buffer += 6;
        }
        for (; i < 20; i++)
        {
            result_buffer[0] = -233;
            result_buffer[1] = -233;
            result_buffer[2] = -233;
            result_buffer[3] = -233;
            result_buffer[4] = -233;
            result_buffer[5] = -233;

            result_buffer += 6;
        }

        rgba_data = 0;

        lock.unlock();

        finish_lock.lock();
        finish_condition.signal();
        finish_lock.unlock();
    }
}

#include <thread>
static std::thread t(worker);

extern "C" {

void yolo_ncnn(const unsigned char* _rgba_data, int _w, int _h, float* _result_buffer)
{
    lock.lock();
    while (rgba_data != 0)
    {
        condition.wait(lock);
    }

    rgba_data = _rgba_data;
    w = _w;
    h = _h;
    result_buffer = _result_buffer;

    lock.unlock();

    condition.signal();

    // wait for finished
    finish_lock.lock();
    while (rgba_data != 0)
    {
        finish_condition.wait(finish_lock);
    }
    finish_lock.unlock();
}

}
