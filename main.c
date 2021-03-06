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
uint INPUT_LINES        = 333;
uint OUTPUT_QUOTES      = 33333;
uint OUTPUT_MOLPS       = 100;  // min output lines per second
uint FAIL_TIMEOUT       = 540;  // fail variance timeout seconds
uint SERVICE_TICK       = 9;    // service tick / poll rate

///

#define FAST_PREDICTABLE_MODE
#define TRAINING_LOOPS 1
uint        _linit      = 1;
float       _lrate      = 0.016325;
float       _ldropout   = 0.130533;
uint        _loptimiser = 2;
float       _lmomentum  = 0.530182;
float       _lrmsalpha  = 0.578107; //0.99
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

    digest = malloc(INPUT_LINES * sizeof(float*));
    if(digest == NULL)
        printf("ERROR malloc() in initMemory() #digest\n");
    for(int i = 0; i < INPUT_LINES; i++)
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
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
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
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
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
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
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

void createPerceptron(ptron* p, const uint weights, const float d)
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

    //const float d = 1/sqrt(p->weights);
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d); //qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = 0; //qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void resetPerceptron(ptron* p, const float d)
{
    //const float d = 1/sqrt(p->weights);
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d); //qRandWeight(-1, 1);
        p->momentum[i] = 0;
    }

    p->bias = 0; //qRandWeight(-1, 1);
    p->bias_momentum = 0;
}

void createPerceptrons()
{
    const uint init_method = _linit;
    float l1d = 1;
    float l2d = 1;
    float l3d = 1;
    float l4d = 1;

    // Xavier uniform
    if(init_method == 1)
    {
        l1d = sqrt(6.0/(FIRSTLAYER_SIZE+HIDDEN_SIZE));
        l2d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l3d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l4d = sqrt(6.0/(HIDDEN_SIZE+1));
    }

    // LeCun uniform
    if(init_method == 2)
    {
        l1d = sqrt(3.0/DIGEST_SIZE);
        l2d = sqrt(3.0/FIRSTLAYER_SIZE);
        l3d = sqrt(3.0/HIDDEN_SIZE);
        l4d = sqrt(3.0/HIDDEN_SIZE);
    }

    // What I thought was LeCun
    if(init_method == 3)
    {
        l1d = pow(DIGEST_SIZE, 0.5);
        l2d = pow(FIRSTLAYER_SIZE, 0.5);
        l3d = pow(HIDDEN_SIZE, 0.5);
        l4d = pow(HIDDEN_SIZE, 0.5);
    }
    
    //printf("%f %f %f %f \n", l1d, l2d, l3d, l4d);

    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE, l1d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE, l2d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE, l3d);
    createPerceptron(&d4, HIDDEN_SIZE, l4d);
}

void resetPerceptrons()
{
    const uint init_method = _linit;
    float l1d = 1;
    float l2d = 1;
    float l3d = 1;
    float l4d = 1;

    // Xavier uniform
    if(init_method == 1)
    {
        l1d = sqrt(6.0/(FIRSTLAYER_SIZE+HIDDEN_SIZE));
        l2d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l3d = sqrt(6.0/(HIDDEN_SIZE+HIDDEN_SIZE));
        l4d = sqrt(6.0/(HIDDEN_SIZE+1));
    }

    // LeCun uniform
    if(init_method == 2)
    {
        l1d = sqrt(3.0/DIGEST_SIZE);
        l2d = sqrt(3.0/FIRSTLAYER_SIZE);
        l3d = sqrt(3.0/HIDDEN_SIZE);
        l4d = sqrt(3.0/HIDDEN_SIZE);
    }

    // What I thought was LeCun
    if(init_method == 3)
    {
        l1d = pow(DIGEST_SIZE, 0.5);
        l2d = pow(FIRSTLAYER_SIZE, 0.5);
        l3d = pow(HIDDEN_SIZE, 0.5);
        l4d = pow(HIDDEN_SIZE, 0.5);
    }

    //printf("%f %f %f %f \n", l1d, l2d, l3d, l4d);

    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        resetPerceptron(&d1[i], l1d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d2[i], l2d);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        resetPerceptron(&d3[i], l3d);
    resetPerceptron(&d4, l4d);
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
    return 1.7159 * tanh(0.666666667 * x);
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

static inline float sigmoidDerivative(float x)
{
    return x * (1 - x);
}

static inline float tanhDerivative(float x)
{
    return 1 - pow(x, 2);
    //return 1-(x*x);
}

//https://stats.stackexchange.com/questions/60166/how-to-use-1-7159-tanh2-3-x-as-activation-function
static inline float lecun_tanhDerivative(float x)
{
    //return 1.14393 * pow((1 / cosh(2*x/3)), 2);
    //return 1.14393 * pow((1 / cosh(x * 0.666666666)), 2);
    const float sx = x * 0.6441272;
    return 1.221595 - (sx*sx);
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
    const float vp = momentum[0];
    const float v = _lmomentum * vp + ( _lrate * error * input );
    const float n = v + _lmomentum * (v - momentum[0]);
    momentum[0] = v;
    return n;
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
        return Nesterov(input, error, momentum);
    else if(_loptimiser == 3)
        return ADAGrad(input, error, momentum);
    else if(_loptimiser == 4)
        return RMSProp(input, error, momentum);
    
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

    const float error = eo - output;

    if(error == 0) // superflous unlikely to happen
        return output;

    float e1[FIRSTLAYER_SIZE];
    float e2[HIDDEN_SIZE];
    float e3[HIDDEN_SIZE];

    // layer 4
    float e4 = _lgain * sigmoidDerivative(output) * error;

    // layer 3 (output)
    float ler = 0;
    for(int j = 0; j < d4.weights; j++)
        ler += d4.data[j] * e4;
    ler += d4.bias * e4;
    
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e3[i] = _lgain * lecun_tanhDerivative(o3[i]) * ler;

    // layer 2
    ler = 0;
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < d3[i].weights; j++)
            ler += d3[i].data[j] * e3[i];
        ler += d3[i].bias * e3[i];
    }
    for(int i = 0; i < HIDDEN_SIZE; i++)
        e2[i] = _lgain * lecun_tanhDerivative(o2[i]) * ler;

    // layer 1
    float k = 0;
    int ki = 0;
    ler = 0;
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        for(int j = 0; j < d2[i].weights; j++)
            ler += d2[i].data[j] * e2[i];
        ler += d2[i].bias * e2[i];
    }
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
    {
        int k0 = 0;
        if(k != 0)
            k0 = 1;
        k += _lgain * lecun_tanhDerivative(o1[i]) * ler;
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
    for(int i = 0; i < INPUT_LINES; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= INPUT_LINES;
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
            if(index == INPUT_LINES)
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
        for(int i = 0; i < INPUT_LINES; i++)
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
                if(count < 16 * OUTPUT_MOLPS)
                    return 0; // if the output rate was less than MOLPS per second, just quit.
                
                count = 0;
                st = time(0);
            }
        }

        fclose(f);
    }

    return 1;
}

float hasFailed(const uint resolution)
{
    int failvariance = 0;
    for(int i = 0; i < 100*resolution; i++)
    {
        const float r = rndScentence();
        if(r < 50)
            failvariance++;
    }
    if(resolution == 1)
        return failvariance;
    else
        return (double)failvariance / (double)resolution;
}

uint huntBestWeights(float* rmse)
{
    *rmse = 0;
    float fv = 0;
    float min = 70;
    const float max = 96.0;
    float highest = 0;
    time_t st = time(0);
    while(fv < min || fv > max) //we want random string to fail at-least 70% of the time / but we don't want it to fail all of the time
    {
        newSRAND(); //kill any predictability in the random generator

        _loptimiser = uRand(0, 4);
        _lrate      = uRandFloat(0.001, 0.03);
        _ldropout   = uRandFloat(0, 0.3);
        if(_loptimiser == 1 || _loptimiser == 2)
            _lmomentum  = uRandFloat(0.1, 0.9);
        if(_loptimiser == 4)
            _lrmsalpha  = uRandFloat(0.2, 0.99);

        resetPerceptrons();
        *rmse = trainDataset();

        fv = hasFailed(100);
        if(fv <= max && fv > highest)
            highest = fv;

        if(time(0) - st > FAIL_TIMEOUT) //If taking longer than 3 mins just settle with the highest logged in that period
        {
            min = highest;
            highest = 0;
            st = time(0);
            printf("Taking too long, new target: %.2f\n", min);
        }

        printf("RMSE: %f / Fail: %.2f\n", *rmse, fv);
    }
    return fv; // fail variance
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init commands
    if(argc > 0)
    {
        DIGEST_SIZE = atoi(argv[1]);
        if(DIGEST_SIZE > DIGEST_SIZE_MAX)
            DIGEST_SIZE = DIGEST_SIZE_MAX;
    }
    if(argc > 1)
        FIRSTLAYER_SIZE =   atoi(argv[2]);
    if(argc > 2)
        HIDDEN_SIZE =       atoi(argv[3]);
    if(argc > 3)
        INPUT_LINES =       atoi(argv[4]);
    if(argc > 4)
        OUTPUT_QUOTES =     atoi(argv[5]);
    if(argc > 5)
        OUTPUT_MOLPS =      atoi(argv[6]);
    if(argc > 6)
        FAIL_TIMEOUT =      atoi(argv[7]);
    if(argc > 7)
        SERVICE_TICK =      atoi(argv[8]);

    // init memory
    initMemory();

    // init discriminator
    createPerceptrons();

    // boot log
    printf("Digest Words: %u\n", DIGEST_SIZE);
    printf("First  Layer: %u\n", FIRSTLAYER_SIZE);
    printf("Hidden Layer: %u\n", HIDDEN_SIZE);
    printf("Digest Lines: %u\n", INPUT_LINES);
    printf("Output Lines: %u\n", OUTPUT_QUOTES);
    printf("Output MOLPS: %u\n", OUTPUT_MOLPS);
    printf("Fail Timeout: %u\n", FAIL_TIMEOUT);
    printf("Service Tick: %u\n\n", SERVICE_TICK);

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines(datasetLocation) >= INPUT_LINES)
        {
            timestamp();
            const time_t st = time(0);
            memset(&wtable, 0x00, TABLE_SIZE_MAX*WORD_SIZE);
            loadTable(dictionaryLocation);
            loadDataset(datasetLocation);
            clearFile(datasetLocation);

            float rmse = 0;
            float fv = huntBestWeights(&rmse);
            while(rndGen(outputLocation, 0.2) == 0)
                fv = huntBestWeights(&rmse);
            
            saveWeights();
            printf("Just generated a new dataset.\n");
            timestamp();
            const double time_taken = ((double)(time(0)-st)) / 60.0;
            printf("Time Taken: %.2f mins\n\n", time_taken);

            FILE* f = fopen(statLocation, "w");
            if(f != NULL)
            {
                fprintf(f, "%f\n", rmse);
                fprintf(f, "%.2f\n", fv);
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

        sleep(SERVICE_TICK);
    }

    // done
    return 0;
}

