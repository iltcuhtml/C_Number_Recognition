#include <time.h>

#define NN_IMPLEMENTATION
#include "NN.h"

#define TRAIN_COUNT 1E6

#define EPS  1E-1
#define RATE 1E-1

typedef struct
{
    Mat x0;

    Mat w1, b1, x1;
    Mat w2, b2, x2;
} XOR_NN;

XOR_NN new_XOR_NN(void)
{
    XOR_NN m;

    m.x0  = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2); mat_rand(m.w1, 0, 1);
    m.b1 = mat_alloc(1, 2); mat_rand(m.b1, 0, 1);
    m.x1 = mat_alloc(1, 2);
    
    m.w2 = mat_alloc(2, 1); mat_rand(m.w2, 0, 1);
    m.b2 = mat_alloc(1, 1); mat_rand(m.b2, 0, 1);
    m.x2 = mat_alloc(1, 1);

    return m;
}

void XOR_forward(XOR_NN m)
{
    mat_dot(m.x1, m.x0, m.w1);
    mat_sum(m.x1, m.b1);
    mat_sig(m.x1);

    mat_dot(m.x2, m.x1, m.w2);
    mat_sum(m.x2, m.b2);
    mat_sig(m.x2);
}

float XOR_cost(XOR_NN m, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == m.x2.cols);
    
    float c = 0.0F;

    for (size_t i = 0; i < ti.rows; ++i)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        
        mat_copy(m.x0, x);
        XOR_forward(m);

        for (size_t ii = 0; ii < to.cols; ++ii)
        {
            float d = MAT_AT(m.x2, 0, ii) - MAT_AT(y, 0, ii);

            c += d * d;
        }
    }

    return c / ti.rows;
}

void finite_diff(XOR_NN m, XOR_NN g, Mat ti, Mat to)
{
    float saved;

    float c = XOR_cost(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; ++i)
        for (size_t ii = 0; ii < m.w1.cols; ++ii)
        {
            saved = MAT_AT(m.w1, i, ii);

            MAT_AT(m.w1, i, ii) += EPS;
            MAT_AT(g.w1, i, ii) = (XOR_cost(m, ti, to) - c) / EPS;
            
            MAT_AT(m.w1, i, ii) = saved;
        }

    for (size_t i = 0; i < m.b1.rows; ++i)
        for (size_t ii = 0; ii < m.b1.cols; ++ii)
        {
            saved = MAT_AT(m.b1, i, ii);

            MAT_AT(m.b1, i, ii) += EPS;
            MAT_AT(g.b1, i, ii) = (XOR_cost(m, ti, to) - c) / EPS;
            
            MAT_AT(m.b1, i, ii) = saved;
        }

    for (size_t i = 0; i < m.w2.rows; ++i)
        for (size_t ii = 0; ii < m.w2.cols; ++ii)
        {
            saved = MAT_AT(m.w2, i, ii);

            MAT_AT(m.w2, i, ii) += EPS;
            MAT_AT(g.w2, i, ii) = (XOR_cost(m, ti, to) - c) / EPS;
            
            MAT_AT(m.w2, i, ii) = saved;
        }

    for (size_t i = 0; i < m.b2.rows; ++i)
        for (size_t ii = 0; ii < m.b2.cols; ++ii)
        {
            saved = MAT_AT(m.b2, i, ii);

            MAT_AT(m.b2, i, ii) += EPS;
            MAT_AT(g.b2, i, ii) = (XOR_cost(m, ti, to) - c) / EPS;
            
            MAT_AT(m.b2, i, ii) = saved;
        }
}

void XOR_train(XOR_NN m, XOR_NN g)
{
    for (size_t i = 0; i < m.w1.rows; ++i)
        for (size_t ii = 0; ii < m.w1.cols; ++ii)
        {
            MAT_AT(m.w1, i, ii) -= MAT_AT(g.w1, i, ii) * RATE;
        }

    for (size_t i = 0; i < m.b1.rows; ++i)
        for (size_t ii = 0; ii < m.b1.cols; ++ii)
        {
            MAT_AT(m.b1, i, ii) -= MAT_AT(g.b1, i, ii) * RATE;
        }

    for (size_t i = 0; i < m.w2.rows; ++i)
        for (size_t ii = 0; ii < m.w2.cols; ++ii)
        {
            MAT_AT(m.w2, i, ii) -= MAT_AT(g.w2, i, ii) * RATE;
        }

    for (size_t i = 0; i < m.b2.rows; ++i)
        for (size_t ii = 0; ii < m.b2.cols; ++ii)
        {
            MAT_AT(m.b2, i, ii) -= MAT_AT(g.b2, i, ii) * RATE;
        }
}

float td[] = {
    0, 0, 0, 
    0, 1, 1, 
    1, 0, 1, 
    1, 1, 0, 
};

int main(void) // int argc, char* argv[]
{
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = {
        .rows = n, 
        .cols = 2, 

        .stride = stride, 

        .es = td
    };

    Mat to = {
        .rows = n, 
        .cols = 1, 

        .stride = stride, 

        .es = td + 2
    };

    XOR_NN m = new_XOR_NN();
    XOR_NN g = new_XOR_NN();
    
    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        finite_diff(m, g, ti, to);
        XOR_train(m, g);
    }

    printf("c = %f\n", XOR_cost(m, ti, to));

    for (size_t i = 0; i < 2; ++i)
        for (size_t ii = 0; ii < 2; ++ii)
        {
            MAT_AT(m.x0, 0, 0) = i;
            MAT_AT(m.x0, 0, 1) = ii;
            
            XOR_forward(m);
            float y = *m.x2.es;

            printf("\n%zu ^ %zu = %f", i, ii, y);
        }
    
    return 0;
}