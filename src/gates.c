#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAIN_COUNT  4
#define REPEAT_COUNT 1E6

#define EPS  1E-1
#define RATE 1E-1

float sigmoidf(float x)
{
    return 1.0F / (1.0F + expf(-x));
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

typedef float sample[3];

// AND-gate
sample AND_gate[TRAIN_COUNT] = {
    {0, 0, 0}, 
    {0, 1, 0}, 
    {1, 0, 0}, 
    {1, 1, 1}, 
};

// OR-gate
sample OR_gate[TRAIN_COUNT] = {
    {0, 0, 0}, 
    {0, 1, 1}, 
    {1, 0, 1}, 
    {1, 1, 1}, 
};

// NAND-gate
sample NAND_gate[TRAIN_COUNT] = {
    {0, 0, 1}, 
    {0, 1, 1}, 
    {1, 0, 1}, 
    {1, 1, 0}, 
};

// NOR-gate
sample NOR_gate[TRAIN_COUNT] = {
    {0, 0, 1}, 
    {0, 1, 0}, 
    {1, 0, 0}, 
    {1, 1, 0}, 
};

// XOR-gate
sample XOR_gate[TRAIN_COUNT] = {
    {0, 0, 0}, 
    {0, 1, 1}, 
    {1, 0, 1}, 
    {1, 1, 0}, 
};

sample *train = XOR_gate;

typedef struct
{
    float N1_w1;
    float N1_w2;
    float N1_b;

    float N2_w1;
    float N2_w2;
    float N2_b;

    float N3_w1;
    float N3_w2;
    float N3_b;
} XOR;

XOR new_XOR(void)
{
    XOR m;

    m.N1_w1 = rand_float();
    m.N1_w2 = rand_float();
    m.N1_b = rand_float();

    m.N2_w1 = rand_float();
    m.N2_w2 = rand_float();
    m.N2_b = rand_float();

    m.N3_w1 = rand_float();
    m.N3_w2 = rand_float();
    m.N3_b = rand_float();

    return m;
}

void print_XOR(XOR m)
{
    printf("N1_w1 = %f, N1_w2 = %f, N1_b = %f\n", m.N1_w1, m.N1_w2, m.N1_b);
    printf("N2_w1 = %f, N2_w2 = %f, N2_b = %f\n", m.N2_w1, m.N2_w2, m.N2_b);
    printf("N3_w1 = %f, N3_w2 = %f, N3_b = %f\n", m.N3_w1, m.N3_w2, m.N3_b);
}

float forward(XOR m, float x1, float x2)
{
    float y1 = sigmoidf(m.N1_w1 * x1 + m.N1_w2 * x2 + m.N1_b);
    float y2 = sigmoidf(m.N2_w1 * x1 + m.N2_w2 * x2 + m.N2_b);

    return sigmoidf(y1 * m.N3_w1 + y2 * m.N3_w2 + m.N3_b);
}

float XOR_cost(XOR m)
{
    float result = 0.0F;

    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y  = forward(m, x1, x2);
        float d  = y - train[i][2];

        result += d * d;
    }

    result /= TRAIN_COUNT;

    return result;
}

XOR XOR_finite_diff(XOR m)
{
    XOR g;

    float c = XOR_cost(m);
    float saved;

    saved = m.N1_w1;
    m.N1_w1 += EPS;
    g.N1_w1 = (XOR_cost(m) - c) / EPS;
    m.N1_w1 = saved;

    saved = m.N1_w2;
    m.N1_w2 += EPS;
    g.N1_w2 = (XOR_cost(m) - c) / EPS;
    m.N1_w2 = saved;

    saved = m.N1_b;
    m.N1_b += EPS;
    g.N1_b = (XOR_cost(m) - c) / EPS;
    m.N1_b = saved;

    saved = m.N2_w1;
    m.N2_w1 += EPS;
    g.N2_w1 = (XOR_cost(m) - c) / EPS;
    m.N2_w1 = saved;

    saved = m.N2_w2;
    m.N2_w2 += EPS;
    g.N2_w2 = (XOR_cost(m) - c) / EPS;
    m.N2_w2 = saved;

    saved = m.N2_b;
    m.N2_b += EPS;
    g.N2_b = (XOR_cost(m) - c) / EPS;
    m.N2_b = saved;

    saved = m.N3_w1;
    m.N3_w1 += EPS;
    g.N3_w1 = (XOR_cost(m) - c) / EPS;
    m.N3_w1 = saved;

    saved = m.N3_w2;
    m.N3_w2 += EPS;
    g.N3_w2 = (XOR_cost(m) - c) / EPS;
    m.N3_w2 = saved;

    saved = m.N3_b;
    m.N3_b += EPS;
    g.N3_b = (XOR_cost(m) - c) / EPS;
    m.N3_b = saved;

    return g;
}

XOR XOR_train(XOR m, XOR g)
{
    m.N1_w1 -= g.N1_w1 * RATE;
    m.N1_w2 -= g.N1_w2 * RATE;
    m.N1_b  -= g.N1_b * RATE;

    m.N2_w1 -= g.N2_w1 * RATE;
    m.N2_w2 -= g.N2_w2 * RATE;
    m.N2_b  -= g.N2_b * RATE;

    m.N3_w1 -= g.N3_w1 * RATE;
    m.N3_w2 -= g.N3_w2 * RATE;
    m.N3_b  -= g.N3_b * RATE;

    return m;
}

int main(void) // int argc, char* argv[]
{
    srand(time(0));

    XOR m = new_XOR();

    for (size_t i = 0; i < REPEAT_COUNT; ++i)
    {
        XOR g = XOR_finite_diff(m);
        m = XOR_train(m, g);
    }

    print_XOR(m);

    printf("\nc = %f\n", XOR_cost(m));

    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];

        printf("\n%f ^ %f = %f", x1, x2, forward(m, x1, x2));
    }

    return 0;
}