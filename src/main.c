#include <time.h>

#define NN_IMPLEMENTATION
#include "NN.h"

int main(void) // int argc, char* argv[]
{
    srand(time(0));

    Mat m = mat_alloc(1, 2);
    mat_rand(m, 5, 10);

    Mat g = mat_alloc(2, 3);
    mat_fill(g, 2);

    Mat dst = mat_alloc(1, 3);
    mat_dot(dst, m, g);

    mat_print(m);
    printf("------------------\n");
    mat_print(dst);

    return 0;
}