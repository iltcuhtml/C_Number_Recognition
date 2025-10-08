#include "NN.h"

// -------------------------
// Utilities
// -------------------------
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float dsigmoid(float y)
{
    return y * (1.0f - y);
}

float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

float drelu(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

// -------------------------
// Matrix
// -------------------------
Mat Mat_alloc(size_t rows, size_t cols)
{
    Mat m = { rows, cols, cols, malloc(sizeof(float) * rows * cols) };

    assert(m.es != NULL);

    return m;
}

void Mat_free(Mat m)
{
    if (m.es)
        free(m.es);
}

void Mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = x;
}

void Mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows && dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
}

void Mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
}

Mat Mat_row(Mat m, size_t row)
{
    return (Mat)
    {
        .rows = 1,
            .cols = m.cols,
            .stride = m.stride,
            .es = &MAT_AT(m, row, 0)
    };
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    assert(m1.cols == m2.rows && dst.rows == m1.rows && dst.cols == m2.cols);

    Mat_fill(dst, 0);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            for (size_t k = 0; k < m1.cols; k++)
                MAT_AT(dst, i, j) += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
}

void Mat_outer(Mat dst, Mat a, Mat b)
{
    assert(a.rows == 1 && b.rows == 1);
    assert(dst.rows == a.cols && dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) = MAT_AT(a, 0, i) * MAT_AT(b, 0, j);
}

void Mat_sum(Mat dst, Mat m)
{
    assert(dst.rows == m.rows && dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
}

void Mat_resize(Mat* m, size_t rows, size_t cols)
{
    if (m->es) free(m->es);

    m->es = malloc(sizeof(float) * rows * cols);

    m->rows = rows;
    m->cols = cols;
    m->stride = cols;
}

// -------------------------
// Transpose a matrix
// -------------------------
Mat Mat_transpose(Mat m)
{
    Mat t = Mat_alloc(m.cols, m.rows);

    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(t, j, i) = MAT_AT(m, i, j);

    return t;
}

void Mat_relu_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            if (MAT_AT(m, i, j) < 0)
                MAT_AT(m, i, j) = 0;
}

void Mat_drelu_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = drelu(MAT_AT(m, i, j));
}

void Mat_dsigmoid_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = dsigmoid(MAT_AT(m, i, j));
}

void Mat_softmax_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        float max_val = -FLT_MAX;

        for (size_t j = 0; j < m.cols; j++)
            if (MAT_AT(m, i, j) > max_val)
                max_val = MAT_AT(m, i, j);

        float sum = 0.0f;

        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - max_val);
            sum += MAT_AT(m, i, j);
        }

        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) /= sum;
    }
}

void Mat_save(FILE* out, Mat m)
{
    fwrite("MATDATA", sizeof(char), 7, out);

    fwrite(&m.rows, sizeof(size_t), 1, out);
    fwrite(&m.cols, sizeof(size_t), 1, out);

    fwrite(m.es, sizeof(float), m.rows * m.cols, out);
}

Mat Mat_load(FILE* in)
{
    char header[7];

    fread(header, sizeof(char), 7, in);
    assert(memcmp(header, "MATDATA", 7) == 0);

    size_t rows, cols;
    fread(&rows, sizeof(size_t), 1, in);
    fread(&cols, sizeof(size_t), 1, in);

    Mat m = Mat_alloc(rows, cols);
    fread(m.es, sizeof(float), rows * cols, in);

    return m;
}

// -------------------------
// Convolution Layer
// -------------------------
ConvLayer Conv_alloc(size_t in_channels, size_t out_channels, size_t kernel_size)
{
    ConvLayer conv = { 0 };

    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel_size = kernel_size;

    conv.kernels = malloc(sizeof(Mat) * in_channels * out_channels);
    conv.biases = malloc(sizeof(Mat) * out_channels);

    for (size_t oc = 0; oc < out_channels; oc++)
    {
        conv.biases[oc] = Mat_alloc(1, 1);
        MAT_AT(conv.biases[oc], 0, 0) = 0;

        for (size_t ic = 0; ic < in_channels; ic++)
            conv.kernels[oc * in_channels + ic] = Mat_alloc(kernel_size, kernel_size);
    }

    return conv;
}

void Conv_free(ConvLayer* conv)
{
    if (!conv)
        return;

    if (conv->kernels)
    {
        for (size_t i = 0; i < conv->out_channels * conv->in_channels; i++)
            Mat_free(conv->kernels[i]);

        free(conv->kernels);
    }

    if (conv->biases)
    {
        for (size_t i = 0; i < conv->out_channels; i++)
            Mat_free(conv->biases[i]);

        free(conv->biases);
    }

    conv->kernels = NULL;
    conv->biases = NULL;
    conv->in_channels = conv->out_channels = conv->kernel_size = 0;
}

// Compute kernel gradient for one input channel and one output channel
// input: H_in x W_in
// d_out: H_out x W_out  (H_out = H_in - k + 1)
// kernel_grad: k x k (assumed allocated & filled zero)
void Conv_compute_kernel_grad(Mat input, Mat d_out, Mat kernel_grad)
{
    size_t k = kernel_grad.rows; // kernel is square

    size_t out_rows = d_out.rows;
    size_t out_cols = d_out.cols;

    Mat_fill(*(&kernel_grad), 0.0f); // ensure kernel_grad zeroed

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float dout = MAT_AT(d_out, y, x);

            for (size_t ky = 0; ky < k; ky++)
                for (size_t kx = 0; kx < k; kx++)
                    MAT_AT(kernel_grad, ky, kx) += MAT_AT(input, y + ky, x + kx) * dout;
        }
}

void Conv_forward_single(Mat kernel, Mat input, Mat* output)
{
    size_t out_rows = input.rows - kernel.rows + 1;
    size_t out_cols = input.cols - kernel.cols + 1;

    *output = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel.rows; ky++)
                for (size_t kx = 0; kx < kernel.cols; kx++)
                    sum += MAT_AT(input, y + ky, x + kx) * MAT_AT(kernel, ky, kx);

            MAT_AT(*output, y, x) = sum;
        }
}

void Conv_forward(ConvLayer conv, Mat* input_channels, Mat* output_channels)
{
    for (size_t oc = 0; oc < conv.out_channels; oc++)
    {
        output_channels[oc] = Mat_alloc(
            input_channels[0].rows - conv.kernel_size + 1,
            input_channels[0].cols - conv.kernel_size + 1
        );

        Mat_fill(output_channels[oc], MAT_AT(conv.biases[oc], 0, 0));

        for (size_t ic = 0; ic < conv.in_channels; ic++)
        {
            Mat temp;

            Conv_forward_single(conv.kernels[oc * conv.in_channels + ic], input_channels[ic], &temp);
            Mat_sum(output_channels[oc], temp);
            Mat_free(temp);
        }

        Mat_relu_inplace(output_channels[oc]);
    }
}

// -------------------------
// CNN + FC full backprop
// -------------------------

// Compute gradient for a single convolution kernel
void Conv_backprop_single(Mat input, Mat d_out, Mat* kernel_grad, Mat* input_grad)
{
    size_t k = kernel_grad->rows; // assume square kernel

    Mat_fill(*kernel_grad, 0);
    Mat_fill(*input_grad, 0);

    size_t out_rows = d_out.rows;
    size_t out_cols = d_out.cols;

    // Gradient w.r.t kernel
    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
            for (size_t ky = 0; ky < k; ky++)
                for (size_t kx = 0; kx < k; kx++)
                    MAT_AT(*kernel_grad, ky, kx) += MAT_AT(input, y + ky, x + kx) * MAT_AT(d_out, y, x);

    // Gradient w.r.t input
    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
            for (size_t ky = 0; ky < k; ky++)
                for (size_t kx = 0; kx < k; kx++)
                    MAT_AT(*input_grad, y + ky, x + kx) += MAT_AT(d_out, y, x) * MAT_AT(*kernel_grad, ky, kx);
}

// -------------------------
// ConvLayer Save / Load
// -------------------------
void Conv_save(FILE* out, ConvLayer conv)
{
    // Write header
    fwrite("CONVLAYR", sizeof(char), 8, out);

    // Write basic parameters
    fwrite(&conv.in_channels, sizeof(size_t), 1, out);
    fwrite(&conv.out_channels, sizeof(size_t), 1, out);
    fwrite(&conv.kernel_size, sizeof(size_t), 1, out);

    // Save all kernels
    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_save(out, conv.kernels[i]);

    // Save all biases
    for (size_t i = 0; i < conv.out_channels; i++)
        Mat_save(out, conv.biases[i]);
}

ConvLayer Conv_load(FILE* in)
{
    char header[8];

    // Read and verify header
    fread(header, sizeof(char), 8, in);
    assert(memcmp(header, "CONVLAYR", 8) == 0);

    ConvLayer conv = { 0 };

    // Read basic parameters
    fread(&conv.in_channels, sizeof(size_t), 1, in);
    fread(&conv.out_channels, sizeof(size_t), 1, in);
    fread(&conv.kernel_size, sizeof(size_t), 1, in);

    // Allocate memory
    conv.kernels = malloc(sizeof(Mat) * conv.in_channels * conv.out_channels);
    conv.biases = malloc(sizeof(Mat) * conv.out_channels);

    // Load all kernels
    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        conv.kernels[i] = Mat_load(in);

    // Load all biases
    for (size_t i = 0; i < conv.out_channels; i++)
        conv.biases[i] = Mat_load(in);

    return conv;
}


// -------------------------
// MaxPool Layer
// -------------------------
Mat MaxPool2D(Mat input, size_t pool_size, size_t stride)
{
    size_t out_rows = (input.rows - pool_size) / stride + 1;
    size_t out_cols = (input.cols - pool_size) / stride + 1;

    Mat out = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float max_val = -FLT_MAX;

            for (size_t py = 0; py < pool_size; py++)
                for (size_t px = 0; px < pool_size; px++)
                {
                    float v = MAT_AT(input, y * stride + py, x * stride + px);

                    if (v > max_val)
                        max_val = v;
                }

            MAT_AT(out, y, x) = max_val;
        }

    return out;
}

// -------------------------
// Flatten Layer
// -------------------------
Mat Flatten(Mat* channels, size_t channel_count)
{
    size_t total = 0;

    for (size_t c = 0; c < channel_count; c++)
        total += channels[c].rows * channels[c].cols;

    Mat out = Mat_alloc(1, total);

    size_t idx = 0;

    for (size_t c = 0; c < channel_count; c++)
        for (size_t i = 0; i < channels[c].rows; i++)
            for (size_t j = 0; j < channels[c].cols; j++)
                MAT_AT(out, 0, idx++) = MAT_AT(channels[c], i, j);

    return out;
}

// -------------------------
// Fully Connected NN
// -------------------------
NN NN_alloc(size_t* arch, size_t arch_count)
{
    assert(arch_count >= 2);

    NN nn = { 0 };

    nn.count = arch_count - 1;

    nn.ws = malloc(nn.count * sizeof(Mat));
    nn.bs = malloc(nn.count * sizeof(Mat));
    nn.as = malloc(arch_count * sizeof(Mat));

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_alloc(arch[i], arch[i + 1]);
        nn.bs[i] = Mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);
    }

    nn.conv_count = 0;
    nn.convs = NULL;

    return nn;
}

void NN_free(NN* nn)
{
    if (!nn)
        return;

    for (size_t i = 0; i < nn->count; i++)
    {
        Mat_free(nn->ws[i]);
        Mat_free(nn->bs[i]);
        Mat_free(nn->as[i + 1]);
    }

    Mat_free(nn->as[0]);

    free(nn->ws);
    free(nn->bs);
    free(nn->as);

    if (nn->convs)
        free(nn->convs);

    nn->ws = nn->bs = nn->as = NULL;
    nn->convs = NULL;
    nn->count = nn->conv_count = 0;
}

void NN_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_rand(nn.ws[i], low, high);
        Mat_rand(nn.bs[i], low, high);
    }
}

// Zero all grads in grad NN
void ZeroGrad(NN grad)
{
    for (size_t i = 0; i < grad.count; i++)
    {
        Mat_fill(grad.ws[i], 0.0f);
        Mat_fill(grad.bs[i], 0.0f);
        Mat_fill(grad.as[i + 1], 0.0f);
    }
    Mat_fill(grad.as[0], 0.0f);
}

void NN_forward(NN nn)
{
    Mat x = NN_INPUT(nn);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_fill(nn.as[i + 1], 0);
        Mat_dot(nn.as[i + 1], x, nn.ws[i]);
        Mat_sum(nn.as[i + 1], nn.bs[i]);
        Mat_relu_inplace(nn.as[i + 1]);

        x = nn.as[i + 1];
    }

    // Apply Softmax to the output of the last layer
    Mat_softmax_inplace(NN_OUTPUT(nn));
}

void NN_xavier_init(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        float limit = sqrtf(6.0f / (nn.ws[i].rows + nn.ws[i].cols));

        for (size_t j = 0; j < nn.ws[i].rows * nn.ws[i].cols; j++)
            nn.ws[i].es[j] = rand_float() * 2.0f * limit - limit;

        Mat_fill(nn.bs[i], 0.0f);
    }
}

// -------------------------
// Compute Cross-Entropy Loss
// -------------------------
float NN_cost(NN nn, Mat inputs, Mat labels)
{
    size_t samples = inputs.rows;
    float cost = 0.0f;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        Mat_copy(NN_INPUT(nn), input_row);
        NN_forward(nn);

        for (size_t j = 0; j < label_row.cols; j++)
        {
            float y = MAT_AT(label_row, 0, j);
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);

            cost -= y * logf(fmaxf(p, 1e-7f));
        }
    }

    return cost / samples;
}

// -------------------------
// Compute accuracy
// -------------------------
float NN_accuracy(NN nn, Mat inputs, Mat labels)
{
    size_t samples = inputs.rows;
    size_t correct = 0;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        Mat_copy(NN_INPUT(nn), input_row);
        NN_forward(nn);

        size_t max_idx = 0;
        float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val)
            {
                max_val = MAT_AT(NN_OUTPUT(nn), 0, j);
                max_idx = j;
            }

        for (size_t j = 0; j < label_row.cols; j++)
            if (MAT_AT(label_row, 0, j) == 1.0f && j == max_idx)
            {
                correct++;
                break;
            }
    }

    return (float)correct / samples;
}

// -------------------------
// Backprop for FC NN only
// -------------------------
void NN_backprop(NN nn, NN grad, Mat inputs, Mat labels)
{
    size_t samples = inputs.rows;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        // Forward pass
        Mat_copy(NN_INPUT(nn), input_row);
        NN_forward(nn);

        // Compute output error (Softmax + CrossEntropy)
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
            MAT_AT(grad.as[nn.count], 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(label_row, 0, j);

        // Backprop through layers
        for (size_t l = nn.count; l-- > 0; )
        {
            Mat* dA = &grad.as[l + 1];
            Mat* A_prev = &nn.as[l];
            Mat* W = &nn.ws[l];
            Mat* dW = &grad.ws[l];
            Mat* db = &grad.bs[l];

            // Gradient w.r.t weights: dW = A_prev^T * dA
            Mat_dot(*dW, *A_prev, *dA);

            // Gradient w.r.t biases: db = dA
            Mat_copy(*db, *dA);

            // Gradient w.r.t previous activation: dA_prev = dA * W^T
            Mat dA_prev = Mat_alloc(A_prev->rows, A_prev->cols);

            Mat_dot(dA_prev, *dA, Mat_transpose(*W));
            Mat_copy(*dA, dA_prev);
            Mat_free(dA_prev);

            // Apply ReLU derivative
            Mat_drelu_inplace(*A_prev);
        }
    }

    // Average gradients over samples
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < grad.ws[i].rows * grad.ws[i].cols; j++)
            grad.ws[i].es[j] /= (float)samples;

        for (size_t j = 0; j < grad.bs[i].rows * grad.bs[i].cols; j++)
            grad.bs[i].es[j] /= (float)samples;
    }
}

// -------------------------
// Gradient descent update
// -------------------------
void NN_learn(NN nn, NN grad, float lr)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows * nn.ws[i].cols; j++)
            nn.ws[i].es[j] -= lr * grad.ws[i].es[j];

        for (size_t j = 0; j < nn.bs[i].rows * nn.bs[i].cols; j++)
            nn.bs[i].es[j] -= lr * grad.bs[i].es[j];
    }
}

// -------------------------
// NN Save / Load
// -------------------------
void NN_save(FILE* out, NN nn)
{
    fwrite("NNMODEL", sizeof(char), 7, out);

    fwrite(&nn.count, sizeof(size_t), 1, out);
    fwrite(&nn.conv_count, sizeof(size_t), 1, out);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_save(out, nn.ws[i]);
        Mat_save(out, nn.bs[i]);
    }

    for (size_t i = 0; i < nn.conv_count; i++)
        Conv_save(out, nn.convs[i]);
}

NN NN_load(FILE* in)
{
    char header[7];

    fread(header, sizeof(char), 7, in);
    assert(memcmp(header, "NNMODEL", 7) == 0);

    NN nn = { 0 };

    fread(&nn.count, sizeof(size_t), 1, in);
    fread(&nn.conv_count, sizeof(size_t), 1, in);

    size_t* arch = malloc(sizeof(size_t) * (nn.count + 1));

    nn.ws = malloc(sizeof(Mat) * nn.count);
    nn.bs = malloc(sizeof(Mat) * nn.count);
    nn.as = malloc(sizeof(Mat) * (nn.count + 1));

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_load(in);
        nn.bs[i] = Mat_load(in);
        arch[i + 1] = nn.ws[i].cols;
    }

    arch[0] = nn.ws[0].rows;
    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);

    if (nn.conv_count > 0)
    {
        nn.convs = malloc(sizeof(ConvLayer) * nn.conv_count);

        for (size_t i = 0; i < nn.conv_count; i++)
            nn.convs[i] = Conv_load(in);
    }

    free(arch);

    return nn;
}

// -------------------------
// Forward for a single sample
// -------------------------
void CNN_forward_sample(NN nn, ConvLayer conv, Mat input_image, Mat* conv_out, Mat* pooled, Mat* flat)
{
    assert(conv_out && pooled && flat);

    // Conv layer forward
    Mat input_channels[1] = { input_image };
    Conv_forward(conv, input_channels, conv_out);

    // MaxPool 2x2 stride 2
    for (size_t c = 0; c < conv.out_channels; c++)
        pooled[c] = MaxPool2D(conv_out[c], 2, 2);

    // Flatten pooled
    *flat = Flatten(pooled, conv.out_channels);

    // Ensure NN input matches
    if (NN_INPUT(nn).cols != flat->cols)
    {
        Mat_resize(&NN_INPUT(nn), 1, flat->cols);
        Mat_fill(NN_INPUT(nn), 0.0f);
    }

    Mat_copy(NN_INPUT(nn), *flat);

    // Forward through FC layers
    NN_forward(nn);
}

// -------------------------
// Backprop for FC NN
// -------------------------
void CNN_backprop_sample(NN nn, NN grad, Mat flat, Mat label)
{
    Mat_copy(NN_INPUT(nn), flat);
    NN_forward(nn);

    // Output error
    for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        MAT_AT(grad.as[nn.count], 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(label, 0, j);

    // Backprop through FC layers
    for (size_t l = nn.count; l-- > 0; )
    {
        Mat* dA = &grad.as[l + 1];
        Mat* A_prev = &nn.as[l];
        Mat* W = &nn.ws[l];
        Mat* dW = &grad.ws[l];
        Mat* db = &grad.bs[l];

        // Gradient w.r.t weights: outer product
        Mat_outer(*dW, *A_prev, *dA);

        // Gradient w.r.t biases
        Mat_copy(*db, *dA);

        // Gradient w.r.t previous activation
        Mat W_T = Mat_transpose(*W);
        Mat dA_prev = Mat_alloc(A_prev->rows, A_prev->cols);

        Mat_dot(dA_prev, *dA, W_T);
        Mat_copy(*dA, dA_prev);
        Mat_free(dA_prev);
        Mat_free(W_T);

        // ReLU derivative
        Mat_drelu_inplace(*A_prev);
    }
}

// -------------------------
// Backprop for one sample (FC + Conv)
// -------------------------
void CNN_backprop_sample_full(NN nn, NN grad_fc, ConvLayer conv, Mat input_image, Mat label)
{
    size_t oc, ic;

    // --- Forward pass ---
    Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

    if (!conv_out || !pooled) { perror("malloc failed"); exit(1); }

    Mat flat;
    CNN_forward_sample(nn, conv, input_image, conv_out, pooled, &flat);

    // --- Fully connected backprop ---
    CNN_backprop_sample(nn, grad_fc, flat, label);

    // --- MaxPool backprop ---
    Mat* d_pooled = malloc(sizeof(Mat) * conv.out_channels);

    if (!d_pooled) { perror("malloc failed"); exit(1); }

    for (oc = 0; oc < conv.out_channels; oc++)
    {
        d_pooled[oc] = Mat_alloc(pooled[oc].rows, pooled[oc].cols);
        Mat_fill(d_pooled[oc], 0);

        // Find max position in 2x2 pooling region
        for (size_t y = 0; y < pooled[oc].rows; y++)
            for (size_t x = 0; x < pooled[oc].cols; x++)
            {
                float max_val = -FLT_MAX;
                size_t max_y = 0, max_x = 0;

                for (size_t py = 0; py < 2; py++)
                    for (size_t px = 0; px < 2; px++)
                    {
                        size_t iy = y * 2 + py;
                        size_t ix = x * 2 + px;

                        if (MAT_AT(conv_out[oc], iy, ix) > max_val)
                        {
                            max_val = MAT_AT(conv_out[oc], iy, ix);
                            max_y = iy;
                            max_x = ix;
                        }
                    }

                // Assign gradient from FC flat backprop
                MAT_AT(d_pooled[oc], y, x) = MAT_AT(flat, 0, y * pooled[oc].cols + x);
            }
    }

    // --- Convolution backprop ---
    for (oc = 0; oc < conv.out_channels; oc++)
    {
        for (ic = 0; ic < conv.in_channels; ic++)
        {
            Mat kernel_grad = Mat_alloc(conv.kernels[0].rows, conv.kernels[0].cols);
            Mat input_grad = Mat_alloc(input_image.rows, input_image.cols);

            Conv_backprop_single(input_image, d_pooled[oc], &kernel_grad, &input_grad);

            // Update kernel weights
            for (size_t k = 0; k < kernel_grad.rows * kernel_grad.cols; k++)
                conv.kernels[oc * conv.in_channels + ic].es[k] -= 0.01f * kernel_grad.es[k]; // lr placeholder

            Mat_free(kernel_grad);
            Mat_free(input_grad);
        }

        // Update bias
        MAT_AT(conv.biases[oc], 0, 0) -= 0.01f; // lr placeholder
    }

    // --- Update fully connected weights ---
    NN_learn(nn, grad_fc, 0.01f); // lr placeholder

    // --- Free allocated memory ---
    Mat_free(flat);

    for (oc = 0; oc < conv.out_channels; oc++)
    {
        Mat_free(conv_out[oc]);
        Mat_free(pooled[oc]);
        Mat_free(d_pooled[oc]);
    }

    free(conv_out);
    free(pooled);
    free(d_pooled);
}

void CNN_update(NN nn, NN grad, float lr)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows * nn.ws[i].cols; j++)
            nn.ws[i].es[j] -= lr * grad.ws[i].es[j];

        for (size_t j = 0; j < nn.bs[i].rows * nn.bs[i].cols; j++)
            nn.bs[i].es[j] -= lr * grad.bs[i].es[j];
    }
}

// -------------------------
// CNN forward + backprop + update for one sample
// -------------------------
void CNN_forward_backprop_update(NN nn, NN grad_fc, ConvLayer* conv, Mat input_image, Mat label, float lr)
{
    // Forward: conv -> relu -> pool -> flatten -> FC
    Mat* conv_out = malloc(sizeof(Mat) * conv->out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv->out_channels);

    if (!conv_out || !pooled) { perror("malloc failed"); exit(1); }

    Mat flat;
    CNN_forward_sample(nn, *conv, input_image, conv_out, pooled, &flat);

    // Zero gradients in grad_fc
    ZeroGrad(grad_fc);

    // Compute FC output error in grad_fc (softmax + cross-entropy)
    for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        MAT_AT(grad_fc.as[nn.count], 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(label, 0, j);

    // Backprop through FC (fills grad_fc.ws, grad_fc.bs and produces grad_fc.as[0] as d(flat))
    CNN_backprop_sample(nn, grad_fc, flat, label);

    // grad_fc.as[0] now contains gradient w.r.t flattened input (shape 1 x flat.cols)
    // Map flattened gradient back to pooled per-channel gradients
    size_t pooled_r = pooled[0].rows;
    size_t pooled_c = pooled[0].cols;
    size_t per_channel_size = pooled_r * pooled_c;

    // For each output channel produce d_pooled (pooled gradient)
    Mat* d_pooled = malloc(sizeof(Mat) * conv->out_channels);

    if (!d_pooled) { perror("malloc failed"); exit(1); }

    for (size_t oc = 0; oc < conv->out_channels; oc++)
    {
        d_pooled[oc] = Mat_alloc(pooled_r, pooled_c);
        Mat_fill(d_pooled[oc], 0.0f);

        for (size_t y = 0; y < pooled_r; y++)
            for (size_t x = 0; x < pooled_c; x++)
            {
                size_t idx = oc * per_channel_size + y * pooled_c + x;
                float g = 0.0f;

                if (idx < grad_fc.as[0].cols)
                    g = MAT_AT(grad_fc.as[0], 0, idx);

                MAT_AT(d_pooled[oc], y, x) = g;
            }
    }

    // MaxPool backprop: expand d_pooled to d_conv_out by placing pooled gradient at max location in 2x2
    Mat* d_conv_out = malloc(sizeof(Mat) * conv->out_channels);

    if (!d_conv_out) { perror("malloc failed"); exit(1); }

    for (size_t oc = 0; oc < conv->out_channels; oc++)
    {
        Mat conv_o = conv_out[oc];
        d_conv_out[oc] = Mat_alloc(conv_o.rows, conv_o.cols);

        Mat_fill(d_conv_out[oc], 0.0f);

        for (size_t py = 0; py < pooled_r; py++)
            for (size_t px = 0; px < pooled_c; px++)
            {
                float grad_val = MAT_AT(d_pooled[oc], py, px);

                // find max position in corresponding 2x2 region
                float maxv = -FLT_MAX;
                size_t max_y = 0, max_x = 0;

                for (size_t ry = 0; ry < 2; ry++)
                    for (size_t rx = 0; rx < 2; rx++)
                    {
                        size_t iy = py * 2 + ry;
                        size_t ix = px * 2 + rx;

                        if (MAT_AT(conv_o, iy, ix) > maxv)
                        {
                            maxv = MAT_AT(conv_o, iy, ix);

                            max_y = iy;
                            max_x = ix;
                        }
                    }


                MAT_AT(d_conv_out[oc], max_y, max_x) = grad_val;
            }
    }

    // Conv backprop: compute kernel gradients per (out_channel, in_channel)
    for (size_t oc = 0; oc < conv->out_channels; oc++)
    {
        for (size_t ic = 0; ic < conv->in_channels; ic++)
        {
            Mat kernel_grad = Mat_alloc(
                conv->kernels[oc * conv->in_channels + ic].rows,
                conv->kernels[oc * conv->in_channels + ic].cols
            );

            Mat_fill(kernel_grad, 0.0f);

            // compute kernel gradient using input_image and d_conv_out[oc]
            Conv_compute_kernel_grad(input_image, d_conv_out[oc], kernel_grad);

            // Update kernel weights (SGD)
            Mat* kernel = &conv->kernels[oc * conv->in_channels + ic];

            for (size_t k = 0; k < kernel->rows * kernel->cols; k++)
                kernel->es[k] -= lr * kernel_grad.es[k];

            Mat_free(kernel_grad);
        }

        // Update bias (sum of d_conv_out elements)
        float bias_grad = 0.0f;

        for (size_t y = 0; y < d_conv_out[oc].rows; y++)
            for (size_t x = 0; x < d_conv_out[oc].cols; x++)
                bias_grad += MAT_AT(d_conv_out[oc], y, x);

        MAT_AT(conv->biases[oc], 0, 0) -= lr * bias_grad;
    }

    // Update FC weights (grad_fc contains dW, db)
    NN_learn(nn, grad_fc, lr);

    // Free temp memory
    for (size_t c = 0; c < conv->out_channels; c++)
    {
        Mat_free(conv_out[c]);
        Mat_free(pooled[c]);
        Mat_free(d_pooled[c]);
        Mat_free(d_conv_out[c]);
    }

    free(conv_out);
    free(pooled);
    free(d_pooled);
    free(d_conv_out);

    Mat_free(flat);
}

// -------------------------
// Train CNN for one epoch
// -------------------------
void CNN_train_epoch(NN nn, NN grad, ConvLayer conv, Mat inputs, Mat labels, float lr)
{
    size_t samples = inputs.rows;

    if (conv.out_channels == 0) return;

    Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

    if (!conv_out || !pooled) { perror("malloc failed"); exit(1); }

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        // Assume square image
        size_t img_size = (size_t)sqrt((double)input_row.cols);

        Mat input_image = Mat_alloc(img_size, img_size);

        for (size_t y = 0; y < img_size; y++)
            for (size_t x = 0; x < img_size; x++)
                MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * img_size + x);

        Mat flat;

        CNN_forward_sample(nn, conv, input_image, conv_out, pooled, &flat);
        CNN_backprop_sample(nn, grad, flat, label_row);
        CNN_update(nn, grad, lr);

        Mat_free(input_image);
        Mat_free(flat);

        for (size_t c = 0; c < conv.out_channels; c++)
        {
            Mat_free(conv_out[c]);
            Mat_free(pooled[c]);
        }
    }

    free(conv_out);
    free(pooled);
}

// -------------------------
// Train CNN for one epoch with Conv+FC update
// -------------------------
void CNN_train_epoch_full(NN nn, NN grad_fc, ConvLayer conv, Mat inputs, Mat labels, float lr, int epoch_num, int total_epochs)
{
    size_t samples = inputs.rows;
    float epoch_cost = 0.0f;
    size_t correct = 0;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        size_t img_size = (size_t)sqrt((double)input_row.cols);
        Mat input_image = Mat_alloc(img_size, img_size);

        for (size_t y = 0; y < img_size; y++)
            for (size_t x = 0; x < img_size; x++)
                MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * img_size + x);

        // --- Forward + Backprop + Update ---
        CNN_forward_backprop_update(nn, grad_fc, &conv, input_image, label_row, lr);

        // --- Compute cost and accuracy ---
        Mat flat;
        Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
        Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

        CNN_forward_sample(nn, conv, input_image, conv_out, pooled, &flat);

        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            float y = MAT_AT(label_row, 0, j);
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);

            epoch_cost -= y * logf(fmaxf(p, 1e-7f));
        }

        size_t max_idx = 0;
        float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val) { max_val = MAT_AT(NN_OUTPUT(nn), 0, j); max_idx = j; }

        for (size_t j = 0; j < label_row.cols; j++)
            if (MAT_AT(label_row, 0, j) == 1.0f && j == max_idx) { correct++; break; }

        // --- Free memory ---
        Mat_free(flat);
        Mat_free(input_image);

        for (size_t c = 0; c < conv.out_channels; c++)
        {
            Mat_free(conv_out[c]);
            Mat_free(pooled[c]);
        }

        free(conv_out);
        free(pooled);
    }

    epoch_cost /= samples;
    float acc = (float)correct / samples;

    printf("Epoch %d/%d, cost = %.4f, accuracy = %.4f\n", epoch_num, total_epochs, epoch_cost, acc);
}

// -------------------------
// Train CNN for one epoch (batch learning, full)
// -------------------------
void CNN_train_epoch_full_batch(NN nn, NN grad_fc, ConvLayer conv, Mat inputs, Mat labels, float lr)
{
    size_t samples = inputs.rows;

    if (conv.out_channels == 0) return;

    Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

    if (!conv_out || !pooled) { perror("malloc failed"); exit(1); }

    float epoch_cost = 0.0f;
    size_t correct = 0;

    for (size_t i = 0; i < samples; i++)
    {
        // --- Prepare input and label ---
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        size_t img_size = (size_t)sqrt((double)input_row.cols);
        Mat input_image = Mat_alloc(img_size, img_size);

        for (size_t y = 0; y < img_size; y++)
            for (size_t x = 0; x < img_size; x++)
                MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * img_size + x);

        // --- Forward pass ---
        Mat flat;
        CNN_forward_sample(nn, conv, input_image, conv_out, pooled, &flat);

        // --- Backpropagation ---
        CNN_backprop_sample_full(nn, grad_fc, conv, input_image, label_row);

        // --- Compute cross-entropy cost ---
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            float y = MAT_AT(label_row, 0, j);
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);

            epoch_cost -= y * logf(fmaxf(p, 1e-7f));
        }

        // --- Compute accuracy ---
        size_t max_idx = 0;
        float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val)
            {
                max_val = MAT_AT(NN_OUTPUT(nn), 0, j);
                max_idx = j;
            }

        for (size_t j = 0; j < label_row.cols; j++)
            if (MAT_AT(label_row, 0, j) == 1.0f && j == max_idx)
            {
                correct++;
                break;
            }

        // --- Free temporary memory ---
        Mat_free(flat);
        Mat_free(input_image);
    }

    // --- Normalize cost and accuracy ---
    epoch_cost /= samples;
    float acc = (float)correct / samples;

    printf("Epoch cost = %.4f, accuracy = %.4f\n", epoch_cost, acc);

    // Free conv outputs memory
    for (size_t c = 0; c < conv.out_channels; c++)
    {
        Mat_free(conv_out[c]);
        Mat_free(pooled[c]);
    }

    free(conv_out);
    free(pooled);
}

// Train epoch with conv+fc updates, prints epoch number, cost, accuracy
void CNN_train_epoch_full_wrapper(NN nn, NN grad_fc, ConvLayer* conv, Mat inputs, Mat labels, float lr, int epoch_num, int total_epochs)
{
    size_t samples = inputs.rows;
    float epoch_cost = 0.0f;
    size_t correct = 0;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat label_row = Mat_row(labels, i);

        size_t img_size = (size_t)sqrt((double)input_row.cols);
        Mat input_image = Mat_alloc(img_size, img_size);

        for (size_t y = 0; y < img_size; y++)
            for (size_t x = 0; x < img_size; x++)
                MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * img_size + x);

        // Forward + backprop + update
        CNN_forward_backprop_update(nn, grad_fc, conv, input_image, label_row, lr);

        // Evaluate to accumulate cost and accuracy (forward again)
        Mat* conv_out = malloc(sizeof(Mat) * conv->out_channels);
        Mat* pooled = malloc(sizeof(Mat) * conv->out_channels);
        Mat flat;

        CNN_forward_sample(nn, *conv, input_image, conv_out, pooled, &flat);

        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            float y = MAT_AT(label_row, 0, j);
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);

            epoch_cost -= y * logf(fmaxf(p, 1e-7f));
        }

        size_t max_idx = 0;
        float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val) { max_val = MAT_AT(NN_OUTPUT(nn), 0, j); max_idx = j; }

        if (MAT_AT(label_row, 0, max_idx) == 1.0f) correct++;

        // free eval temp
        Mat_free(flat);

        for (size_t c = 0; c < conv->out_channels; c++)
        {
            Mat_free(conv_out[c]);
            Mat_free(pooled[c]);
        }

        free(conv_out);
        free(pooled);

        Mat_free(input_image);
    }

    epoch_cost /= samples;
    float acc = (float)correct / samples;

    printf("Epoch %d/%d, cost = %.4f, accuracy = %.4f\n", epoch_num, total_epochs, epoch_cost, acc);
}

// -------------------------
// Save CNN + FC NN
// -------------------------
void CNN_save(FILE* file, ConvLayer conv, NN nn)
{
    if (!file)
        return;

    fwrite("CNNMODEL", sizeof(char), 8, file);

    fwrite(&conv.in_channels, sizeof(size_t), 1, file);
    fwrite(&conv.out_channels, sizeof(size_t), 1, file);
    fwrite(&conv.kernel_size, sizeof(size_t), 1, file);

    // Save kernels (1D array)
    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_save(file, conv.kernels[i]);

    // Save biases
    for (size_t i = 0; i < conv.out_channels; i++)
        Mat_save(file, conv.biases[i]);

    // Save FC NN
    NN_save(file, nn);
}

// -------------------------
// Load CNN + FC NN
// -------------------------
void CNN_load(ConvLayer* conv, NN* nn, FILE* file)
{
    if (!file)
        return;

    char header[8];
    fread(header, sizeof(char), 8, file);

    if (memcmp(header, "CNNMODEL", 8) != 0)
    {
        fprintf(stderr, "Invalid CNN model file\n");

        return;
    }

    fread(&conv->in_channels, sizeof(size_t), 1, file);
    fread(&conv->out_channels, sizeof(size_t), 1, file);
    fread(&conv->kernel_size, sizeof(size_t), 1, file);

    size_t kernel_count = conv->out_channels * conv->in_channels;

    conv->kernels = malloc(sizeof(Mat) * kernel_count);
    conv->biases = malloc(sizeof(Mat) * conv->out_channels);

    for (size_t i = 0; i < kernel_count; i++)
        conv->kernels[i] = Mat_load(file);

    for (size_t i = 0; i < conv->out_channels; i++)
        conv->biases[i] = Mat_load(file);

    *nn = NN_load(file);
}