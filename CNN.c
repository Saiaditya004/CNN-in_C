#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define be32toh(x) ((((x) & 0xff000000) >> 24) | \
                    (((x) & 0x00ff0000) >> 8) |  \
                    (((x) & 0x0000ff00) << 8) |  \
                    (((x) & 0x000000ff) << 24))
#else
#include <endian.h>
#endif

#define IMAGE_SIZE 28
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define NUM_FILTERS 8
#define NUM_CLASSES 10
#define TEST_SAMPLES 1000
#define TRAIN_SAMPLES 60000
#define EPOCHS 5
#define LEARNING_RATE 0.01
#define BATCH_SIZE 32

typedef struct
{
    double filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
    double gradients[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
} Conv3x3;

typedef struct
{
    double *weights;
    double *biases;
    double *gradients;
    double *bias_gradients;
} Softmax;

typedef struct
{
    unsigned char *train_images;
    unsigned char *train_labels;
    unsigned char *test_images;
    unsigned char *test_labels;
} MNIST;

typedef struct
{
    FILE *fp;
    uint32_t record_size;
    uint32_t total_records;
    uint32_t *indices;
    uint8_t *batch_data;
    int batch_size;
    int current_batch;
} BatchLoader;

typedef struct
{
    size_t total_allocated;
    size_t peak_allocated;
    size_t allocations;
    size_t deallocations;
} MemoryStats;

MemoryStats mem_stats = {0, 0, 0, 0};

void *tracked_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr)
    {

        size_t *size_ptr = (size_t *)malloc(sizeof(size_t) + size);
        if (!size_ptr)
        {
            free(ptr);
            return NULL;
        }
        *size_ptr = size;
        ptr = size_ptr + 1;

        mem_stats.total_allocated += size;
        mem_stats.allocations++;
        if (mem_stats.total_allocated > mem_stats.peak_allocated)
        {
            mem_stats.peak_allocated = mem_stats.total_allocated;
        }
    }
    return ptr;
}

void tracked_free(void *ptr)
{
    if (ptr)
    {

        size_t *size_ptr = ((size_t *)ptr) - 1;
        size_t size = *size_ptr;

        mem_stats.total_allocated -= size;
        mem_stats.deallocations++;
        free(size_ptr);
    }
}

void print_memory_stats()
{
    fprintf(stderr, "Memory Usage:\n");
    fprintf(stderr, "  Current: %.2f MB\n", mem_stats.total_allocated / (1024.0 * 1024.0));
    fprintf(stderr, "  Peak:    %.2f MB\n", mem_stats.peak_allocated / (1024.0 * 1024.0));
    fprintf(stderr, "  Allocs:  %zu\n", mem_stats.allocations);
    fprintf(stderr, "  Frees:   %zu\n", mem_stats.deallocations);
    fprintf(stderr, "  Leaks:   %zu\n", mem_stats.allocations - mem_stats.deallocations);
}

void *safe_malloc(size_t size, const char *error_msg)
{
    void *ptr = malloc(size);
    if (!ptr)
    {
        fprintf(stderr, "Memory allocation failed: %s (%zu bytes)\n",
                error_msg, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t size, const char *error_msg)
{
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0)
    {
        fprintf(stderr, "Memory reallocation failed: %s (%zu bytes)\n",
                error_msg, size);
        free(ptr);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

FILE *safe_fopen(const char *filename, const char *mode, const char *error_msg)
{
    FILE *fp = fopen(filename, mode);
    if (!fp)
    {
        fprintf(stderr, "File operation failed: %s (%s)\n",
                error_msg, filename);
        exit(EXIT_FAILURE);
    }
    return fp;
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

void get_memory_info(size_t *used, size_t *available)
{
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    *used = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
    *available = memInfo.ullAvailPhys;
}

#else
#include <sys/resource.h>

void increase_stack_size(size_t stack_size)
{
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < stack_size)
        {
            rl.rlim_cur = stack_size;
            if (rl.rlim_max < rl.rlim_cur)
            {
                rl.rlim_max = rl.rlim_cur;
            }
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "Warning: Could not increase stack size\n");
            }
        }
    }
}

#include <unistd.h>
void get_memory_info(size_t *used, size_t *available)
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long avail_pages = sysconf(_SC_AVPHYS_PAGES);

    *used = (pages - avail_pages) * page_size;
    *available = avail_pages * page_size;
}
#endif

int allocate_mnist_memory(MNIST *mnist);
void free_mnist_memory(MNIST *mnist);
void free_softmax_memory(Softmax *softmax);
void train(MNIST *mnist, Conv3x3 *conv, Softmax *softmax);
void evaluate(MNIST *mnist, Conv3x3 *conv, Softmax *softmax);

BatchLoader *BatchLoader_create(const char *filename, int batch_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    struct
    {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
    } header;

    if (fread(&header, sizeof(header), 1, fp) != 1)
    {
        fprintf(stderr, "Failed to read header from: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    if (header.magic != 0 || header.type != 0x08)
    {
        fprintf(stderr, "Invalid file format: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    uint32_t *dims = (uint32_t *)malloc(header.ndims * sizeof(uint32_t));
    if (!dims)
    {
        fprintf(stderr, "Memory allocation failed for dimensions\n");
        fclose(fp);
        return NULL;
    }

    if (fread(dims, sizeof(uint32_t), header.ndims, fp) != header.ndims)
    {
        fprintf(stderr, "Failed to read dimensions from: %s\n", filename);
        free(dims);
        fclose(fp);
        return NULL;
    }

    uint32_t total_records = be32toh(dims[0]);

    uint32_t record_size = 1;
    for (int i = 1; i < header.ndims; i++)
    {
        record_size *= be32toh(dims[i]);
    }

    free(dims);

    BatchLoader *loader = (BatchLoader *)malloc(sizeof(BatchLoader));
    if (!loader)
    {
        fprintf(stderr, "Failed to allocate memory for batch loader\n");
        fclose(fp);
        return NULL;
    }

    loader->fp = fp;
    loader->record_size = record_size;
    loader->total_records = total_records;
    loader->batch_size = batch_size;
    loader->current_batch = -1;

    loader->indices = (uint32_t *)malloc(total_records * sizeof(uint32_t));
    if (!loader->indices)
    {
        fprintf(stderr, "Failed to allocate memory for indices\n");
        free(loader);
        fclose(fp);
        return NULL;
    }

    for (uint32_t i = 0; i < total_records; i++)
    {
        loader->indices[i] = i;
    }

    loader->batch_data = (uint8_t *)malloc(batch_size * record_size);
    if (!loader->batch_data)
    {
        fprintf(stderr, "Failed to allocate memory for batch data\n");
        free(loader->indices);
        free(loader);
        fclose(fp);
        return NULL;
    }

    return loader;
}

int BatchLoader_next_batch(BatchLoader *loader)
{
    if (!loader)
        return 0;

    loader->current_batch++;

    int start_idx = loader->current_batch * loader->batch_size;
    if (start_idx >= loader->total_records)
    {

        loader->current_batch = 0;
        start_idx = 0;

        for (uint32_t i = loader->total_records - 1; i > 0; i--)
        {
            uint32_t j = rand() % (i + 1);
            uint32_t temp = loader->indices[i];
            loader->indices[i] = loader->indices[j];
            loader->indices[j] = temp;
        }
    }

    int records_to_load = loader->batch_size;
    if (start_idx + records_to_load > loader->total_records)
    {
        records_to_load = loader->total_records - start_idx;
    }

    long header_size = 4 + 4;

    for (int i = 0; i < records_to_load; i++)
    {

        uint32_t idx = loader->indices[start_idx + i];
        long offset = header_size + idx * loader->record_size;
        fseek(loader->fp, offset, SEEK_SET);

        if (fread(loader->batch_data + i * loader->record_size, 1,
                  loader->record_size, loader->fp) != loader->record_size)
        {
            fprintf(stderr, "Error reading record %u\n", idx);
        }
    }

    return records_to_load;
}

void BatchLoader_destroy(BatchLoader *loader)
{
    if (!loader)
        return;

    if (loader->fp)
    {
        fclose(loader->fp);
    }

    if (loader->indices)
    {
        free(loader->indices);
    }

    if (loader->batch_data)
    {
        free(loader->batch_data);
    }

    free(loader);
}

void evaluate(MNIST *mnist, Conv3x3 *conv, Softmax *softmax);

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

int load_mnist(MNIST *mnist,
               const char *train_image_path,
               const char *train_label_path,
               const char *test_image_path,
               const char *test_label_path)
{
    FILE *train_image_file = fopen(train_image_path, "rb");
    if (!train_image_file)
    {
        printf("Error opening training images file: %s\n", train_image_path);
        return 0;
    }

    FILE *train_label_file = fopen(train_label_path, "rb");
    if (!train_label_file)
    {
        printf("Error opening training labels file: %s\n", train_label_path);
        fclose(train_image_file);
        return 0;
    }

    FILE *test_image_file = fopen(test_image_path, "rb");
    if (!test_image_file)
    {
        printf("Error opening test images file: %s\n", test_image_path);
        fclose(train_image_file);
        fclose(train_label_file);
        return 0;
    }

    FILE *test_label_file = fopen(test_label_path, "rb");
    if (!test_label_file)
    {
        printf("Error opening test labels file: %s\n", test_label_path);
        fclose(train_image_file);
        fclose(train_label_file);
        fclose(test_image_file);
        return 0;
    }

    fseek(train_image_file, 16, SEEK_SET);
    fseek(train_label_file, 8, SEEK_SET);
    fseek(test_image_file, 16, SEEK_SET);
    fseek(test_label_file, 8, SEEK_SET);

    if (!allocate_mnist_memory(mnist))
    {
        fclose(train_image_file);
        fclose(train_label_file);
        fclose(test_image_file);
        fclose(test_label_file);
        return 0;
    }

    fread(mnist->train_images, 1, TRAIN_SAMPLES * IMAGE_SIZE * IMAGE_SIZE, train_image_file);
    fread(mnist->train_labels, 1, TRAIN_SAMPLES, train_label_file);

    fread(mnist->test_images, 1, TEST_SAMPLES * IMAGE_SIZE * IMAGE_SIZE, test_image_file);
    fread(mnist->test_labels, 1, TEST_SAMPLES, test_label_file);

    fclose(train_image_file);
    fclose(train_label_file);
    fclose(test_image_file);
    fclose(test_label_file);

    return 1;
}

void init_conv3x3(Conv3x3 *conv)
{
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            for (int j = 0; j < FILTER_SIZE; j++)
            {

                double scale = sqrt(6.0 / (9 + 9));
                conv->filters[f][i][j] = ((double)rand() / RAND_MAX * 2 - 1) * scale;
                conv->gradients[f][i][j] = 0;
            }
        }
    }
}

int init_softmax(Softmax *softmax)
{
    int input_size = 13 * 13 * 8;
    double scale = sqrt(6.0 / (input_size + NUM_CLASSES));

    softmax->weights = (double *)malloc(input_size * NUM_CLASSES * sizeof(double));
    if (!softmax->weights)
    {
        printf("Failed to allocate memory for softmax weights\n");
        return 0;
    }

    softmax->biases = (double *)calloc(NUM_CLASSES, sizeof(double));
    if (!softmax->biases)
    {
        printf("Failed to allocate memory for softmax biases\n");
        free(softmax->weights);
        return 0;
    }

    softmax->gradients = (double *)calloc(input_size * NUM_CLASSES, sizeof(double));
    if (!softmax->gradients)
    {
        printf("Failed to allocate memory for softmax gradients\n");
        free(softmax->weights);
        free(softmax->biases);
        return 0;
    }

    softmax->bias_gradients = (double *)calloc(NUM_CLASSES, sizeof(double));
    if (!softmax->bias_gradients)
    {
        printf("Failed to allocate memory for softmax bias gradients\n");
        free(softmax->weights);
        free(softmax->biases);
        free(softmax->gradients);
        return 0;
    }

    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            softmax->weights[i * NUM_CLASSES + j] = ((double)rand() / RAND_MAX * 2 - 1) * scale;
        }
    }

    return 1;
}

void conv3x3_forward(double input[IMAGE_SIZE][IMAGE_SIZE],
                     Conv3x3 *conv,
                     double output[26][26][NUM_FILTERS],
                     double pre_activation[26][26][NUM_FILTERS])
{
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < 26; i++)
        {
            for (int j = 0; j < 26; j++)
            {
                double sum = 0;
                for (int k = 0; k < FILTER_SIZE; k++)
                {
                    for (int l = 0; l < FILTER_SIZE; l++)
                    {
                        sum += input[i + k][j + l] * conv->filters[f][k][l];
                    }
                }
                pre_activation[i][j][f] = sum;
                output[i][j][f] = relu(sum);
            }
        }
    }
}

void maxpool2_forward(double input[26][26][NUM_FILTERS],
                      double output[13][13][NUM_FILTERS],
                      int indices[13][13][NUM_FILTERS][2])
{
    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                double max_val = -1e9;
                int max_i = 0, max_j = 0;

                for (int k = 0; k < 2; k++)
                {
                    for (int l = 0; l < 2; l++)
                    {
                        double val = input[i * 2 + k][j * 2 + l][f];
                        if (val > max_val)
                        {
                            max_val = val;
                            max_i = k;
                            max_j = l;
                        }
                    }
                }
                output[i][j][f] = max_val;
                indices[i][j][f][0] = max_i;
                indices[i][j][f][1] = max_j;
            }
        }
    }
}

void softmax_forward(double input[13][13][NUM_FILTERS],
                     Softmax *softmax,
                     double output[NUM_CLASSES])
{
    double totals[NUM_CLASSES] = {0};
    double exp_sum = 0;

    for (int j = 0; j < NUM_CLASSES; j++)
    {
        for (int i = 0; i < 13 * 13 * 8; i++)
        {
            totals[j] += ((double *)input)[i] * softmax->weights[i * NUM_CLASSES + j];
        }
        totals[j] += softmax->biases[j];
        output[j] = exp(totals[j]);
        exp_sum += output[j];
    }

    for (int j = 0; j < NUM_CLASSES; j++)
    {
        output[j] /= exp_sum;
    }
}

void softmax_backward(double input[13][13][NUM_FILTERS],
                      double d_output[NUM_CLASSES],
                      Softmax *softmax,
                      double d_input[13][13][NUM_FILTERS])
{

    memset(d_input, 0, sizeof(double) * 13 * 13 * NUM_FILTERS);

    for (int i = 0; i < 13 * 13 * 8; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            softmax->gradients[i * NUM_CLASSES + j] += ((double *)input)[i] * d_output[j];
            softmax->bias_gradients[j] += d_output[j];
            ((double *)d_input)[i] += softmax->weights[i * NUM_CLASSES + j] * d_output[j];
        }
    }
}

void maxpool2_backward(double d_output[13][13][NUM_FILTERS],
                       int indices[13][13][NUM_FILTERS][2],
                       double d_input[26][26][NUM_FILTERS])
{
    memset(d_input, 0, sizeof(double) * 26 * 26 * NUM_FILTERS);

    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int j = 0; j < 13; j++)
            {
                int k = indices[i][j][f][0];
                int l = indices[i][j][f][1];
                d_input[i * 2 + k][j * 2 + l][f] = d_output[i][j][f];
            }
        }
    }
}

void conv3x3_backward(double d_output[26][26][NUM_FILTERS],
                      double input[IMAGE_SIZE][IMAGE_SIZE],
                      double pre_activation[26][26][NUM_FILTERS],
                      Conv3x3 *conv)
{

    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < 26; i++)
        {
            for (int j = 0; j < 26; j++)
            {
                d_output[i][j][f] *= relu_derivative(pre_activation[i][j][f]);
            }
        }
    }

    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int k = 0; k < FILTER_SIZE; k++)
        {
            for (int l = 0; l < FILTER_SIZE; l++)
            {
                double grad = 0;
                for (int i = 0; i < 26; i++)
                {
                    for (int j = 0; j < 26; j++)
                    {
                        grad += input[i + k][j + l] * d_output[i][j][f];
                    }
                }
                conv->gradients[f][k][l] += grad;
            }
        }
    }
}

void update_parameters(Conv3x3 *conv, Softmax *softmax, int batch_size)
{

    for (int f = 0; f < NUM_FILTERS; f++)
    {
        for (int i = 0; i < FILTER_SIZE; i++)
        {
            for (int j = 0; j < FILTER_SIZE; j++)
            {
                conv->filters[f][i][j] -= LEARNING_RATE * (conv->gradients[f][i][j] / batch_size);
                conv->gradients[f][i][j] = 0;
            }
        }
    }

    for (int i = 0; i < 13 * 13 * 8; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            softmax->weights[i * NUM_CLASSES + j] -= LEARNING_RATE * (softmax->gradients[i * NUM_CLASSES + j] / batch_size);
            softmax->gradients[i * NUM_CLASSES + j] = 0;
        }
    }

    for (int j = 0; j < NUM_CLASSES; j++)
    {
        softmax->biases[j] -= LEARNING_RATE * (softmax->bias_gradients[j] / batch_size);
        softmax->bias_gradients[j] = 0;
    }
}

void train(MNIST *mnist, Conv3x3 *conv, Softmax *softmax)
{

    double image[IMAGE_SIZE][IMAGE_SIZE];
    double conv_out[26][26][NUM_FILTERS];
    double pre_activation[26][26][NUM_FILTERS];
    double pool_out[13][13][NUM_FILTERS];
    double softmax_out[NUM_CLASSES];
    int pool_indices[13][13][NUM_FILTERS][2];

    double d_softmax_out[NUM_CLASSES];
    double d_pool_out[13][13][NUM_FILTERS];
    double d_conv_out[26][26][NUM_FILTERS];

    printf("Starting training for %d epochs...\n", EPOCHS);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
        int correct = 0;
        double total_loss = 0;

        int indices[TRAIN_SAMPLES];
        for (int i = 0; i < TRAIN_SAMPLES; i++)
            indices[i] = i;

        for (int i = 0; i < TRAIN_SAMPLES; i++)
        {
            int j = rand() % TRAIN_SAMPLES;
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        for (int batch_start = 0; batch_start < TRAIN_SAMPLES; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > TRAIN_SAMPLES)
                batch_end = TRAIN_SAMPLES;
            int batch_size = batch_end - batch_start;

            for (int b = 0; b < batch_size; b++)
            {
                int idx = indices[batch_start + b];

                for (int x = 0; x < IMAGE_SIZE; x++)
                {
                    for (int y = 0; y < IMAGE_SIZE; y++)
                    {
                        image[x][y] = (mnist->train_images[idx * IMAGE_SIZE * IMAGE_SIZE + x * IMAGE_SIZE + y] / 255.0) - 0.5;
                    }
                }

                conv3x3_forward(image, conv, conv_out, pre_activation);
                maxpool2_forward(conv_out, pool_out, pool_indices);
                softmax_forward(pool_out, softmax, softmax_out);

                int label = mnist->train_labels[idx];
                double loss = -log(softmax_out[label] > 1e-10 ? softmax_out[label] : 1e-10);
                total_loss += loss;

                int prediction = 0;
                double max_prob = softmax_out[0];
                for (int j = 1; j < NUM_CLASSES; j++)
                {
                    if (softmax_out[j] > max_prob)
                    {
                        max_prob = softmax_out[j];
                        prediction = j;
                    }
                }

                if (prediction == label)
                    correct++;

                for (int j = 0; j < NUM_CLASSES; j++)
                {
                    d_softmax_out[j] = softmax_out[j];
                }
                d_softmax_out[label] -= 1.0;

                softmax_backward(pool_out, d_softmax_out, softmax, d_pool_out);
                maxpool2_backward(d_pool_out, pool_indices, d_conv_out);
                conv3x3_backward(d_conv_out, image, pre_activation, conv);
            }

            update_parameters(conv, softmax, batch_size);

            if ((batch_start / BATCH_SIZE) % 100 == 99)
            {
                printf("  Batch %d/%d: Loss=%.4f, Accuracy=%.2f%%\n",
                       batch_start / BATCH_SIZE + 1,
                       (TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE,
                       total_loss / (batch_start + batch_size),
                       100.0 * correct / (batch_start + batch_size));
            }
        }

        evaluate(mnist, conv, softmax);
    }
}

void train_with_batches(Conv3x3 *conv, Softmax *softmax,
                        const char *images_file, const char *labels_file,
                        int epochs, int batch_size, double learning_rate)
{
    BatchLoader *images = BatchLoader_create(images_file, batch_size);
    BatchLoader *labels = BatchLoader_create(labels_file, batch_size);

    if (!images || !labels)
    {
        fprintf(stderr, "Failed to create batch loaders\n");
        if (images)
            BatchLoader_destroy(images);
        if (labels)
            BatchLoader_destroy(labels);
        return;
    }

    double image[IMAGE_SIZE][IMAGE_SIZE];
    double conv_out[26][26][NUM_FILTERS];
    double pre_activation[26][26][NUM_FILTERS];
    double pool_out[13][13][NUM_FILTERS];
    double softmax_out[NUM_CLASSES];
    int pool_indices[13][13][NUM_FILTERS][2];

    double d_softmax_out[NUM_CLASSES];
    double d_pool_out[13][13][NUM_FILTERS];
    double d_conv_out[26][26][NUM_FILTERS];

    printf("Starting training for %d epochs with batch size %d...\n", epochs, batch_size);

    double total_loss = 0;
    int correct = 0;
    int total = 0;
    int batches_per_epoch = (images->total_records + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        total_loss = 0;
        correct = 0;
        total = 0;

        for (int batch = 0; batch < batches_per_epoch; batch++)
        {
            int records = BatchLoader_next_batch(images);
            BatchLoader_next_batch(labels);

            for (int i = 0; i < records; i++)
            {

                for (int x = 0; x < IMAGE_SIZE; x++)
                {
                    for (int y = 0; y < IMAGE_SIZE; y++)
                    {
                        image[x][y] = (images->batch_data[i * IMAGE_SIZE * IMAGE_SIZE + x * IMAGE_SIZE + y] / 255.0) - 0.5;
                    }
                }

                conv3x3_forward(image, conv, conv_out, pre_activation);
                maxpool2_forward(conv_out, pool_out, pool_indices);
                softmax_forward(pool_out, softmax, softmax_out);

                int label = labels->batch_data[i];
                double loss = -log(softmax_out[label] > 1e-10 ? softmax_out[label] : 1e-10);
                total_loss += loss;

                int prediction = 0;
                double max_prob = softmax_out[0];
                for (int j = 1; j < NUM_CLASSES; j++)
                {
                    if (softmax_out[j] > max_prob)
                    {
                        max_prob = softmax_out[j];
                        prediction = j;
                    }
                }

                if (prediction == label)
                    correct++;
                total++;

                for (int j = 0; j < NUM_CLASSES; j++)
                {
                    d_softmax_out[j] = softmax_out[j];
                }
                d_softmax_out[label] -= 1.0;

                softmax_backward(pool_out, d_softmax_out, softmax, d_pool_out);
                maxpool2_backward(d_pool_out, pool_indices, d_conv_out);
                conv3x3_backward(d_conv_out, image, pre_activation, conv);
            }

            update_parameters(conv, softmax, records);

            if (batch % 10 == 0)
            {
                printf("  Batch %d/%d: Loss=%.4f, Accuracy=%.2f%%\n",
                       batch + 1, batches_per_epoch,
                       total_loss / (total > 0 ? total : 1),
                       100.0 * correct / (total > 0 ? total : 1));

                print_memory_stats();
            }
        }

        printf("Epoch %d/%d: Loss=%.4f, Accuracy=%.2f%%\n",
               epoch + 1, epochs,
               total_loss / (total > 0 ? total : 1),
               100.0 * correct / (total > 0 ? total : 1));
    }

    BatchLoader_destroy(images);
    BatchLoader_destroy(labels);
}

void evaluate(MNIST *mnist, Conv3x3 *conv, Softmax *softmax)
{
    int correct = 0;
    double total_loss = 0;

    printf("Evaluating on test set...\n");

    for (int i = 0; i < TEST_SAMPLES; i++)
    {
        double image[IMAGE_SIZE][IMAGE_SIZE];
        for (int x = 0; x < IMAGE_SIZE; x++)
        {
            for (int y = 0; y < IMAGE_SIZE; y++)
            {
                image[x][y] = (mnist->test_images[i * IMAGE_SIZE * IMAGE_SIZE + x * IMAGE_SIZE + y] / 255.0) - 0.5;
            }
        }

        double conv_out[26][26][NUM_FILTERS];
        double pre_activation[26][26][NUM_FILTERS];
        double pool_out[13][13][NUM_FILTERS];
        double softmax_out[NUM_CLASSES];
        int pool_indices[13][13][NUM_FILTERS][2];

        conv3x3_forward(image, conv, conv_out, pre_activation);
        maxpool2_forward(conv_out, pool_out, pool_indices);
        softmax_forward(pool_out, softmax, softmax_out);

        int label = mnist->test_labels[i];
        double loss = -log(softmax_out[label] > 1e-10 ? softmax_out[label] : 1e-10);
        total_loss += loss;

        int prediction = 0;
        double max_prob = softmax_out[0];
        for (int j = 1; j < NUM_CLASSES; j++)
        {
            if (softmax_out[j] > max_prob)
            {
                max_prob = softmax_out[j];
                prediction = j;
            }
        }

        if (prediction == label)
            correct++;

        if ((i + 1) % 100 == 0)
        {
            printf("  Processed %d/%d test samples\n", i + 1, TEST_SAMPLES);
        }
    }

    printf("Test results: Loss=%.4f, Accuracy=%.2f%%\n",
           total_loss / TEST_SAMPLES,
           100.0 * correct / TEST_SAMPLES);
}

int allocate_mnist_memory(MNIST *mnist)
{

    mnist->train_images = (unsigned char *)malloc(TRAIN_SAMPLES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char));
    if (!mnist->train_images)
    {
        printf("Failed to allocate memory for training images\n");
        return 0;
    }

    mnist->train_labels = (unsigned char *)malloc(TRAIN_SAMPLES * sizeof(unsigned char));
    if (!mnist->train_labels)
    {
        printf("Failed to allocate memory for training labels\n");
        free(mnist->train_images);
        return 0;
    }

    mnist->test_images = (unsigned char *)malloc(TEST_SAMPLES * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char));
    if (!mnist->test_images)
    {
        printf("Failed to allocate memory for test images\n");
        free(mnist->train_images);
        free(mnist->train_labels);
        return 0;
    }

    mnist->test_labels = (unsigned char *)malloc(TEST_SAMPLES * sizeof(unsigned char));
    if (!mnist->test_labels)
    {
        printf("Failed to allocate memory for test labels\n");
        free(mnist->train_images);
        free(mnist->train_labels);
        free(mnist->test_images);
        return 0;
    }

    return 1;
}

void free_mnist_memory(MNIST *mnist)
{
    if (mnist->train_images)
        free(mnist->train_images);
    if (mnist->train_labels)
        free(mnist->train_labels);
    if (mnist->test_images)
        free(mnist->test_images);
    if (mnist->test_labels)
        free(mnist->test_labels);

    mnist->train_images = NULL;
    mnist->train_labels = NULL;
    mnist->test_images = NULL;
    mnist->test_labels = NULL;
}

void free_softmax_memory(Softmax *softmax)
{
    if (softmax->weights)
        free(softmax->weights);
    if (softmax->biases)
        free(softmax->biases);
    if (softmax->gradients)
        free(softmax->gradients);
    if (softmax->bias_gradients)
        free(softmax->bias_gradients);

    softmax->weights = NULL;
    softmax->biases = NULL;
    softmax->gradients = NULL;
    softmax->bias_gradients = NULL;
}

int main()
{
    MNIST mnist = {NULL, NULL, NULL, NULL};
    Softmax softmax = {NULL, NULL, NULL, NULL};
    Conv3x3 conv;

    const char *train_images = "train-images.idx3-ubyte";
    const char *train_labels = "train-labels.idx1-ubyte";
    const char *test_images = "t10k-images.idx3-ubyte";
    const char *test_labels = "t10k-labels.idx1-ubyte";

    FILE *f1 = fopen(train_images, "rb");
    FILE *f2 = fopen(train_labels, "rb");
    FILE *f3 = fopen(test_images, "rb");
    FILE *f4 = fopen(test_labels, "rb");

    if (!f1 || !f2 || !f3 || !f4)
    {
        printf("MNIST dataset files not found in current directory.\n");
        printf("Please make sure the following files are present:\n");
        printf("- %s %s\n", train_images, f1 ? "✓" : "✗");
        printf("- %s %s\n", train_labels, f2 ? "✓" : "✗");
        printf("- %s %s\n", test_images, f3 ? "✓" : "✗");
        printf("- %s %s\n", test_labels, f4 ? "✓" : "✗");

        if (f1)
            fclose(f1);
        if (f2)
            fclose(f2);
        if (f3)
            fclose(f3);
        if (f4)
            fclose(f4);

        return 1;
    }

    fclose(f1);
    fclose(f2);
    fclose(f3);
    fclose(f4);

    if (!allocate_mnist_memory(&mnist))
    {
        printf("Failed to allocate memory for MNIST data\n");
        return 1;
    }

    if (!load_mnist(&mnist, train_images, train_labels, test_images, test_labels))
    {
        free_mnist_memory(&mnist);
        return 1;
    }

    init_conv3x3(&conv);

    if (!init_softmax(&softmax))
    {
        free_mnist_memory(&mnist);
        return 1;
    }

    printf("MNIST CNN initialized with dataset loaded!\n");
    printf("Training samples: %d\n", TRAIN_SAMPLES);
    printf("Test samples: %d\n", TEST_SAMPLES);

    int use_batch_processing = 0;

    if (use_batch_processing)
    {
        printf("Using batch processing for training...\n");
        train_with_batches(&conv, &softmax, train_images, train_labels,
                           EPOCHS, BATCH_SIZE, LEARNING_RATE);
    }
    else
    {
        printf("Using standard training method...\n");
        train(&mnist, &conv, &softmax);
    }

    evaluate(&mnist, &conv, &softmax);

    free_mnist_memory(&mnist);
    free_softmax_memory(&softmax);

    print_memory_stats();

    return 0;
}