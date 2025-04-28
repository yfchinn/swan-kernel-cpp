#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
#include <ap_int.h>

#define MAX_DATA_SIZE 256
#define WIDEN_FACTOR 8

static void load_vec(float *i_vec, hls::stream<float> &inStream, int vec_size)
{
mem_rd:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE II = 1
    inStream << i_vec[i];
  }
}

static void compute_softmax(hls::stream<float> &in_stream,
                            hls::stream<float> &out_stream, int vec_size)
{
  int in_max_idx = vec_size;
  float sum = 0;
  float vec_local_1[MAX_DATA_SIZE];
  float vec_local_2[MAX_DATA_SIZE];
#pragma HLS ARRAY_PARTITION variable = vec_local_1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = vec_local_2 cyclic factor = 16
#pragma HLS dataflow

  if (vec_size == -1)
  {
    in_max_idx = 256;
  }

  for (int i = 0; i < in_max_idx; i++)
  {
    vec_local_1[i] = in_stream.read();
  }

  // 1. Get Max
  float max_val = vec_local_1[0];
get_max:
  for (int i = 1; i < in_max_idx; i++)
  {
#pragma HLS PIPELINE II = 1
    if (vec_local_1[i] > max_val)
    {
      max_val = vec_local_1[i];
    }
  }

  // 2. Exp and Sum
calc_exp:
  for (int i = 0; i < in_max_idx; i++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    vec_local_2[i] = hls::expf(vec_local_1[i] - max_val);
  }

sum_exp:
  for (int i = 0; i < in_max_idx; i++)
  {
#pragma HLS PIPELINE off
    sum += vec_local_2[i];
  }

// 3. Normalize
normalize:
  for (int i = 0; i < in_max_idx; i++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    vec_local_2[i] /= sum;
  }
// Output the result
write_out:
  for (int i = 0; i < in_max_idx; i++)
  {
#pragma HLS PIPELINE II = 1
    out_stream << vec_local_2[i];
  }
}

static void store_result(float *out, hls::stream<float> &out_stream,
                         int vec_size)
{
mem_wr:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE II = 1
    out[i] = out_stream.read();
  }
}

extern "C"
{
  void kernel_softmax(float *i_vec, float *o_vec, int vec_size)
  {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0 max_read_burst_length = 256 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0 max_write_burst_length = 256 num_write_outstanding = 16

#pragma HLS INTERFACE s_axilite port = i_vec bundle = control
#pragma HLS INTERFACE s_axilite port = o_vec bundle = control
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static hls::stream<float> vec_stream("vec_stream");
#pragma HLS STREAM variable = vec_stream depth = 16
    static hls::stream<float> out_stream("out_stream");
#pragma HLS STREAM variable = out_stream depth = 16

#pragma HLS dataflow
    load_vec(i_vec, vec_stream, vec_size);
    compute_softmax(vec_stream, out_stream, vec_size);
    store_result(o_vec, out_stream, vec_size);
  }
}

#endif // USE_CPU_ONLY
