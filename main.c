/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
    https://github.com/DaisyGAN/
--------------------------------------------------
    DaisyGANv6

    Technically not a generative adversarial network anymore.

    v6 aims to turn DaisyGAN into a modular component that
    can be re-used to create multi-process neural net models.
*/

#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wformat-zero-length"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <locale.h>

#define uint uint32_t
#define NO_LEARN -2

///

#define DIGEST_SIZE_MAX 16
#define TABLE_SIZE_MAX  80000
#define WORD_SIZE       256
#define MESSAGE_SIZE    WORD_SIZE*DIGEST_SIZE
uint DIGEST_SIZE        = 16;
uint FIRSTLAYER_SIZE    = 128;
uint HIDDEN_SIZE        = 128;
uint DATA_SIZE          = 333;
uint OUTPUT_QUOTES      = 33333;

///

#define FAST_PREDICTABLE_MODE
#define TRAINING_LOOPS 1
float       _lrate      = 0.03;
float       _ldropout   = 0.2;
uint        _loptimiser = 1;
float       _lmomentum  = 0.1;
float       _lrmsalpha  = 0.2; //0.99
const float _lgain      = 1.0;

//

struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// discriminator 
ptron* d1;
ptron* d2;
ptron* d3;
ptron d4;

// normalised training data
float** digest;

//word lookup table / index
char wtable[TABLE_SIZE_MAX][WORD_SIZE] = {0};
uint TABLE_SIZE = 0;
uint TABLE_SIZE_H = 0;

// module id
const char dictionaryLocation[] = "/var/www/html/botdict.txt";
const char datasetLocation[]    = "botmsg.txt";
const char outputLocation[]     = "out.txt";
const char statLocation[]       = "stat.txt";
const char weightLocation[]     = "weights.dat";


//*************************************
// utility functions
//*************************************

void initMemory()
{
    d1 = malloc(FIRSTLAYER_SIZE * sizeof(ptron));
    if(d1 == NULL)
        printf("ERROR malloc() in initMemory() #d1\n");

    d2 = malloc(HIDDEN_SIZE * sizeof(ptron));
    if(d2 == NULL)
        printf("ERROR malloc() in initMemory() #d2\n");

    d3 = malloc(HIDDEN_SIZE * sizeof(ptron));
    if(d3 == NULL)
        printf("ERROR malloc() in initMemory() #d3\n");

    digest = malloc(DATA_SIZE * sizeof(float*));
    if(digest == NULL)
        printf("ERROR malloc() in initMemory() #digest\n");
    for(int i = 0; i < DATA_SIZE; i++)
    {
        digest[i] = malloc(DIGEST_SIZE * sizeof(float));
        if(digest[i] == NULL)
            printf("ERROR malloc() in initMemory() #digest[%u]\n", i);
    }
}

void loadTable(const char* file)
{
    FILE* f = fopen(file, "r");
    if(f)
    {
        uint index = 0;
        while(fgets(wtable[index], WORD_SIZE, f) != NULL)
        {
            char* pos = strchr(wtable[index], '\n');
            if(pos != NULL)
                *pos = '\0';
            
            index++;
            if(index == TABLE_SIZE_MAX)
                break;
        }
        TABLE_SIZE = index;
        TABLE_SIZE_H = TABLE_SIZE / 2;
        fclose(f);
    }
}

float getWordNorm(const char* word)
{
    for(uint i = 0; i < TABLE_SIZE; i++)
        if(strcmp(word, wtable[i]) == 0)
            return (((double)i) / (double)(TABLE_SIZE_H))-1.0;

    return 0;
}

void saveWeights()
{
    FILE* f = fopen(weightLocation, "w");
    if(f != NULL)
    {
        for(uint i = 0; i < FIRSTLAYER_SIZE; i++)
        {
            if(fwrite(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #2w\n");
            
            if(fwrite(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #2m\n");

            if(fwrite(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #2w\n");
            
            if(fwrite(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #2m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #3w\n");
            
            if(fwrite(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #3m\n");

            if(fwrite(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #3w\n");
            
            if(fwrite(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #3m\n");
        }

        if(fwrite(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #4w\n");
        
        if(fwrite(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #4m\n");

        if(fwrite(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #4w\n");
        
        if(fwrite(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #4m\n");

        fclose(f);
    }
}

void loadWeights()
{
    FILE* f = fopen(weightLocation, "r");
    if(f == NULL)
    {
        printf("!!! no pre-existing weights where found, starting from random initialisation.\n\n\n-----------------\n");
        return;
    }

    for(uint i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        while(fread(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    while(fread(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

    while(fread(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
        sleep(333);

    fclose(f);
}

float qRandFloat(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / RAND_MAX) * (max-min) ) + min;
}

float uRandFloat(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandFloat(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / RAND_MAX) * (max-min) ) + min;
#endif
}

float qRandWeight(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRandWeight(min, max);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
#endif
}

uint qRand(const uint min, const uint umax)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}

uint uRand(const uint min, const uint umax)
{
#ifdef FAST_PREDICTABLE_MODE
    return qRand(min, umax);
#else
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
#endif
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srand(time(0)+c.tv_nsec);
}

//https://stackoverflow.com/questions/30432856/best-way-to-get-number-of-lines-in-a-file-c
uint countLines(const char* file)
{
    uint lines = 0;
    FILE *fp = fopen(file, "r");
    if(fp != NULL)
    {
        while(EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
            ++lines;
        
        fclose(fp);
    }
    return lines;
}

void clearFile(const char* file)
{
    FILE *f = fopen(file, "w");
    if(f != NULL)
    {
        fprintf(f, "");
        fclose(f);
    }
}

void timestamp()
{
    const time_t ltime = time(0);
    printf("%s", asctime(localtime(&ltime)));
}


//*************************************
// create layer
//*************************************

void createPerceptron(ptron* p, const uint weights)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
    {
        printf("Perceptron creation failed (w)%u.\n", weights);
        return;
    }

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        printf("Perceptron creation failed (m)%u.\n", weights);
        return;
    }

    p->weights = weights;

    for(uint i = 0; i < weights; i++)
    {
        p->data[i] = qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void resetPerceptron(ptron* p)
{
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void createPerceptrons()
{
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE);
    createPerceptron(&d4, HIDDEN_SIZE);
}

void resetPerceptrons()
{
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        resetPerceptron(&d1[i]);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d2[i]);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d3[i]);
    resetPerceptron(&d4);
}


//*************************************
// activation functions
// https://en.wikipedia.org/wiki/Activation_function
// https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
// https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
//*************************************

static inline float bipolarSigmoid(float x)
{
    return (1 - exp(-x)) / (1 + exp(-x));
}

static inline float fbiSigmoid(float x)
{
    return (1 - fabs(x)) / (1 + fabs(x));
}

static inline float arctan(float x)
{
    return atan(x);
}

static inline float lecun_tanh(float x)
{
    return 1.7159 * atan(0.666666667 * x);
}

static inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static inline float fSigmoid(float x)
{
    return x / (1 + fabs(x));
    //return 0.5 * (x / (1 + abs(x))) + 0.5;
}

static inline float swish(float x)
{
    return x * sigmoid(x);
}

static inline float leakyReLU(float x)
{
    if(x < 0){x *= 0.01;}
    return x;
}

static inline float ReLU(float x)
{
    if(x < 0){x = 0;}
    return x;
}

static inline float ReLU6(float x)
{
    if(x < 0){x = 0;}
    if(x > 6){x = 6;}
    return x;
}

static inline float leakyReLU6(float x)
{
    if(x < 0){x *= 0.01;}
    if(x > 6){x = 6;}
    return x;
}

static inline float smoothReLU(float x) //aka softplus
{
    return log(1 + exp(x));
}

static inline float logit(float x)
{
    return log(x / (1 - x));
}

void softmax_transform(float* w, const uint32_t n)
{
    float d = 0;
    for(size_t i = 0; i < n; i++)
        d += exp(w[i]);

    for(size_t i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

float crossEntropy(const float predicted, const float expected) //log loss
{
    if(expected == 1)
      return -log(predicted);
    else
      return -log(1 - predicted);
}

float doPerceptron(const float* in, ptron* p)
{
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

    return ro;
}

static inline float SGD(const float input, const float error)
{
    return _lrate * error * input;
}

float Momentum(const float input, const float error, float* momentum)
{
    // const float err = (_lrate * error * input);
    // const float ret = err + _lmomentum * momentum[0];
    // momentum[0] = err;
    // return ret;

    const float err = (_lrate * error * input) + _lmomentum * momentum[0];
    momentum[0] = err;
    return err;
}

float Nesterov(const float input, const float error, float* momentum)
{
    // const float ret = _lrate * ( input * (input + _lmomentum * momentum[0]) ) + (_lmomentum * momentum[0]);
    // momentum[0] = input;
    // return ret;

    const float ret = _lrate * ( error * (input + _lmomentum * momentum[0]) ) + (_lmomentum * momentum[0]);
    momentum[0] = input;
    return ret;
}

float ADAGrad(const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] += err * err;
    return (_lrate / sqrt(momentum[0] + 1e-8)) * err; // 0.00000001
}

float RMSProp(const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] = _lrmsalpha * momentum[0] + (1 - _lrmsalpha) * (err * err);
    return (_lrate / sqrt(momentum[0] + 1e-8)) * err; // 0.00000001
}

float Optional(const float input, const float error, float* momentum)
{
    if(_loptimiser == 1)
        return Momentum(input, error, momentum);
    else if(_loptimiser == 2)
        return ADAGrad(input, error, momentum);
    else if(_loptimiser == 3)
        return RMSProp(input, error, momentum);
    else if(_loptimiser == 4)
        return Nesterov(input, error, momentum);
    
    return SGD(input, error);
}


//*************************************
// network training functions
//*************************************

float doDiscriminator(const float* input, const float eo)
{
/**************************************
    Forward Prop
**************************************/

    // layer one, inputs (fc)
    float o1[FIRSTLAYER_SIZE];
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1[i] = lecun_tanh(doPerceptron(input, &d1[i]));

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = lecun_tanh(doPerceptron(&o1[0], &d2[i]));

    // layer three, hidden (fc)
    float o3[HIDDEN_SIZE];
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3[i] = lecun_tanh(doPerceptron(&o2[0], &d3[i]));

    // layer four, output (fc compression)
    const float output = sigmoid(lecun_tanh(doPerceptron(&o3[0], &d4)));

    if(eo == NO_LEARN)
        return output;

/**************************************
    Backward Prop Error
**************************************/

    float e1[FIRSTLAYER_SIZE];
    float e2[HIDDEN_SIZE];
    float e3[HIDDEN_SIZE];

    // layer 4
    const float error = eo - output;
    float e4 = _lgain * output * (1-output) * error;

    // layer 3 (output)
    float ler = 0;
    for(int j = 0; j < d4.weights; j++)
        ler += d4.data[j] * e4;
    ler += d4.bias * e4;
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e3[i] = _lgain * o3[i] * (1-o3[i]) * ler;

    // layer 2
    ler = 0;
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < d3[i].weights; j++)
            ler += d3[i].data[j] * e3[i];
        ler += d3[i].bias * e3[i];
        
        e2[i] = _lgain * o2[i] * (1-o2[i]) * ler;
    }

    // layer 1
    ler = 0;
    float k = 0;
    int ki = 0;
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        for(int j = 0; j < d2[i].weights; j++)
            ler += d2[i].data[j] * e2[i];
        ler += d2[i].bias * e2[i];
        
        int k0 = 0;
        if(k != 0)
            k0 = 1;
        k += _lgain * o1[i] * (1-o1[i]) * ler;
        if(k0 == 1)
        {
            e1[ki] = k / 2;
            ki++;
        }
    }

/**************************************
    Update Weights
**************************************/

    // layer 1
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;

        for(int j = 0; j < d1[i].weights; j++)
            d1[i].data[j] += Optional(input[j], e1[i], &d1[i].momentum[j]); //SGD(input[j], e1[i]); //Momentum(input[j], e1[i], &d1[i].momentum[j]);

        d1[i].bias += Optional(1, e1[i], &d1[i].bias_momentum); //SGD(1, e1[i]); //Momentum(1, e1[i], &d1[i].bias_momentum);
    }

    // layer 2
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;

        for(int j = 0; j < d2[i].weights; j++)
            d2[i].data[j] += Optional(o1[j], e2[i], &d2[i].momentum[j]); //SGD(o1[j], e2[i]); //Momentum(o1[j], e2[i], &d2[i].momentum[j]);

        d2[i].bias += Optional(1, e2[i], &d2[i].bias_momentum); //SGD(1, e2[i]); //Momentum(1, e2[i], &d2[i].bias_momentum);
    }

    // layer 3
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        if(_ldropout != 0 && uRandFloat(0, 1) <= _ldropout)
            continue;
            
        for(int j = 0; j < d3[i].weights; j++)
            d3[i].data[j] += Optional(o2[j], e3[i], &d3[i].momentum[j]); //SGD(o2[j], e3[i]); //Momentum(o2[j], e3[i], &d3[i].momentum[j]);

        d3[i].bias += Optional(1, e3[i], &d3[i].bias_momentum); //SGD(1, e3[i]); //Momentum(1, e3[i], &d3[i].bias_momentum);
    }

    // layer 4
    for(int j = 0; j < d4.weights; j++)
        d4.data[j] += Optional(o3[j], e4, &d4.momentum[j]); //SGD(o3[j], e4); //Momentum(o3[j], e4, &d4.momentum[j]);

    d4.bias += Optional(1, e4, &d4.bias_momentum); //SGD(1, e4); //Momentum(1, e4, &d4.bias_momentum);

    // done, return forward prop output
    return output;
}

float rmseDiscriminator()
{
    float squaremean = 0;
    for(int i = 0; i < DATA_SIZE; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void loadDataset(const char* file)
{
    // read training data
    FILE* f = fopen(file, "r");
    if(f)
    {
        char line[MESSAGE_SIZE];
        uint index = 0;
        while(fgets(line, MESSAGE_SIZE, f) != NULL)
        {
            char* pos = strchr(line, '\n');
            if(pos != NULL)
                *pos = '\0';
            uint i = 0;
            char* w = strtok(line, " ");
            
            while(w != NULL)
            {
                digest[index][i] = getWordNorm(w); //normalise
                w = strtok(NULL, " ");
                i++;
            }

            index++;
            if(index == DATA_SIZE)
                break;
        }
        fclose(f);
    }

    printf("Training Data Loaded.\n");
}

float trainDataset()
{
    float rmse = 0;

    // train discriminator
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = 0; i < DATA_SIZE; i++)
        {
            // train discriminator on data
            doDiscriminator(&digest[i][0], 1);

            // detrain discriminator on random word sequences 
            float output[DIGEST_SIZE_MAX] = {0};
            const int len = uRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                output[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //uRandWeight(-1, 1);
            doDiscriminator(&output[0], 0);
        }

        rmse = rmseDiscriminator();
    }

    // return rmse
    return rmse;
}


//*************************************
// program functions
//*************************************

float rndScentence()
{
    float nstr[DIGEST_SIZE_MAX] = {0};
    const int len = uRand(1, DIGEST_SIZE);
    for(int i = 0; i < len; i++)
        nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //qRandFloat(-1, 1)

    const float r = doDiscriminator(nstr, NO_LEARN);
    return r*100;
}

uint rndGen(const char* file, const float max)
{
    FILE* f = fopen(file, "w");
    if(f != NULL)
    {
        uint count = 0;
        time_t st = time(0);
        for(int k = 0; k < OUTPUT_QUOTES; NULL)
        {
            float nstr[DIGEST_SIZE_MAX] = {0};
            const int len = uRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //qRandFloat(-1, 1)

            const float r = doDiscriminator(nstr, NO_LEARN);
            if(1-r < max)
            {
                for(int i = 0; i < DIGEST_SIZE; i++)
                {
                    const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
                    if(nstr[i] != 0)
                        fprintf(f, "%s ", wtable[ind]);
                }
                
                k++;
                count++;
                fprintf(f, "\n");
            }

            if(time(0) - st > 16) // after 16 seconds
            {
                if(count < 16*100)
                    return 0; // if the output rate was less than 100 per second, just quit.
                
                count = 0;
                st = time(0);
            }
        }

        fclose(f);
    }

    return 1;
}

float findBest(const uint maxopt)
{
    float lowest_low = 999999999;
    for(uint i = 0; i <= maxopt; i++)
    {
        _loptimiser = i;

        for(uint j = 0; j < 3; j++)
        {
            resetPerceptrons();
            const float rmse = trainDataset();
            if(rmse > 0 && rmse < lowest_low)
            {
                lowest_low = rmse;
                saveWeights();
            }
        }
    }
    return lowest_low;
}

uint hasFailed()
{
    int failvariance = 0;
    for(int i = 0; i < 100; i++)
    {
        const float r = rndScentence();
        if(r < 50)
            failvariance++;
    }
    return failvariance;
}

float rmse = 0;
uint fv = 0;
void huntBestWeights()
{
    fv = 0;
    rmse = 0;
    uint min = 70;
    const uint max = 95;
    uint highest = 0;
    time_t st = time(0);
    while(fv < min || fv > max) //we want random string to fail at-least 70% of the time / but we don't want it to fail all of the time
    {
        newSRAND(); //kill any predictability in the random generator

        _lrate      = uRandFloat(0.001, 0.03);
        _ldropout   = uRandFloat(0.2, 0.3);
        _lmomentum  = uRandFloat(0.1, 0.9);
        _lrmsalpha  = uRandFloat(0.2, 0.99);

        rmse = findBest(1);

        loadWeights();
        fv = hasFailed();
        if(fv <= max && fv > highest)
            highest = fv;

        if(time(0) - st > 180) //If taking longer than 3 mins just settle with the highest logged in that period
        {
            min = highest;
            highest = 0;
            st = time(0);
            printf("Taking too long, new target: %u\n", min);
        }

        printf("RMSE: %f / Fail: %u\n", rmse, fv);
    }
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init command
    if(argc == 6)
    {
        DIGEST_SIZE = atoi(argv[1]);
        if(DIGEST_SIZE > DIGEST_SIZE_MAX)
            DIGEST_SIZE = DIGEST_SIZE_MAX;
        FIRSTLAYER_SIZE = atoi(argv[2]);
        HIDDEN_SIZE = atoi(argv[3]);
        DATA_SIZE = atoi(argv[4]);
        OUTPUT_QUOTES = atoi(argv[5]);
    }

    // init memory
    initMemory();

    // init discriminator
    createPerceptrons();

    // boot log
    printf("Digest Words: %u\n", DIGEST_SIZE);
    printf("First  Layer: %u\n", FIRSTLAYER_SIZE);
    printf("Hidden Layer: %u\n", HIDDEN_SIZE);
    printf("Digest Lines: %u\n", DATA_SIZE);
    printf("Output Lines: %u\n\n", OUTPUT_QUOTES);

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines(datasetLocation) >= DATA_SIZE)
        {
            timestamp();
            const time_t st = time(0);
            memset(&wtable, 0x00, TABLE_SIZE_MAX*WORD_SIZE);
            loadTable(dictionaryLocation);
            loadDataset(datasetLocation);
            clearFile(datasetLocation);

            _lrate      = uRandFloat(0.001, 0.03);
            _ldropout   = uRandFloat(0.2, 0.3);
            _lmomentum  = uRandFloat(0.1, 0.9);
            _lrmsalpha  = uRandFloat(0.2, 0.99);

            huntBestWeights();
            while(rndGen(outputLocation, 0.1) == 0)
                huntBestWeights();
            
            printf("Just generated a new dataset.\n");
            timestamp();
            const double time_taken = ((double)(time(0)-st)) / 60.0;
            printf("Time Taken: %.2f mins\n\n", time_taken);

            FILE* f = fopen(statLocation, "w");
            if(f != NULL)
            {
                fprintf(f, "%f\n", rmse);
                fprintf(f, "%u\n", fv);
                fprintf(f, "%.2f\n", time_taken);
                fprintf(f, "%f\n", _lrate);
                fprintf(f, "%f\n", _ldropout);
                fprintf(f, "%f\n", _lmomentum);
                fprintf(f, "%f\n", _lrmsalpha);
                fprintf(f, "%u\n", FIRSTLAYER_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1);
                fprintf(f, "%u\n", FIRSTLAYER_SIZE*(DIGEST_SIZE+1) + HIDDEN_SIZE*(FIRSTLAYER_SIZE+1) + HIDDEN_SIZE*(HIDDEN_SIZE+1) + (HIDDEN_SIZE+1));
                const time_t ltime = time(0);
                fprintf(f, "%s\n", asctime(localtime(&ltime)));
                fclose(f);
            }
        }

        sleep(9);
    }

    // done
    return 0;
}

