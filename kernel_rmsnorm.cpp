#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

static void load_vec(float *i_vec, hls::stream<float> &inStream, int vec_size)
{
mem_rd:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE
    inStream << i_vec[i];
  }
}

static void compute_rmsnorm(hls::stream<float> &in1_stream,
                            hls::stream<float> &in2_stream,
                            hls::stream<float> &out_stream)
{

  float vec_local_1[288];
  float vec_local_2[288];
#pragma HLS ARRAY_PARTITION variable = vec_local_1 cyclic factor = 64
#pragma HLS ARRAY_PARTITION variable = vec_local_2 cyclic factor = 64
  float sum_local = 0;
  float tmp_sum[288] = {0}; // 288 is ...
  float tmp_result[288];
#pragma HLS ARRAY_PARTITION variable = tmp_sum complete
#pragma HLS ARRAY_PARTITION variable = tmp_result complete
// #pragma HLS dataflow

load_1:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE
    vec_local_1[i] = in1_stream.read();
  }

load_2:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE
    vec_local_2[i] = in2_stream.read();
  }

sum_square_1:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    tmp_sum[i] = vec_local_1[i] * vec_local_1[i];
  }
sum_square_2:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS UNROLL
    sum_local += tmp_sum[i];
  }
  constexpr float eps = 1e-5;
  const float norm = 1 / std::sqrt(sum_local / 288 + eps);

compute_out_mult:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 64
    tmp_result[i] = vec_local_1[i] * norm * vec_local_2[i];
  }

compute_out_write:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE
    out_stream << tmp_result[i];
  }
}

static void store_result(float *out, hls::stream<float> &out_stream,
                         int vec_size)
{
mem_wr:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE
    out[i] = out_stream.read();
  }
}

extern "C"
{
  void kernel_rmsnorm(float *i_vec_1, float *i_vec_2, float *o_vec,
                      int vec_size)
  {
#pragma HLS INTERFACE m_axi port = i_vec_1 bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = i_vec_2 bundle = gmem1 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0 max_write_burst_length = 256
#pragma HLS INTERFACE s_axilite port = i_vec_1 bundle = control
#pragma HLS INTERFACE s_axilite port = i_vec_2 bundle = control
#pragma HLS INTERFACE s_axilite port = o_vec bundle = control
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static hls::stream<float> vec_stream_1("vec_stream_1");
#pragma HLS STREAM variable = vec_stream_1 depth = 512
    static hls::stream<float> vec_stream_2("mat_stream_2");
#pragma HLS STREAM variable = vec_stream_2 depth = 512
    static hls::stream<float> out_stream("out_stream");
#pragma HLS STREAM variable = out_stream depth = 512

#pragma HLS dataflow
    load_vec(i_vec_1, vec_stream_1, vec_size);
    load_vec(i_vec_2, vec_stream_2, vec_size);
    compute_rmsnorm(vec_stream_1, vec_stream_2, out_stream);
    store_result(o_vec, out_stream, vec_size);
  }
}

#endif // USE_CPU_ONLY
