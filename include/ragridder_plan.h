#ifndef __VISIBILITY_H__
#define __VISIBILITY_H__

#include "datatype.h"

struct visibility
{
    /*
    visbility related componment
    u,v,w - coordinate in frequent domain, unit wavelength
    vis - complex number
    weight
    */
    PCS *u;
    PCS *v;
    PCS *w;

    CPX *vis;
    PCS *weight;
    bool *flag;

    //PCS time;
    PCS *frequency;
    int pirange; // in pirange do nothing, not in pirange *2PI
    int sign;
    //int antenna_1;
    //int antenna_2;
    //PCS image_weight;
    //PCS channel_bandwidth;
    //PCS integration_time;
};

struct ragridder_plan
{
    /* data */
    visibility kv;
    PCS fov;
    PCS pixelsize_x;
    PCS pixelsize_y;
    PCS w_max;
    PCS w_min;
    PCS w_s_r; //w term scaling ratio
    int num_w;
    PCS w_0;
    PCS delta_w;
    int speedoflight;
    CPX *dirty_image;
    int width;
    int height;
    int channel;
    int nrow;
    int cur_channel;   // current channel
    int w_term_method; // 0 for w-stacking, 1 for improved w-stacking
};

#endif
