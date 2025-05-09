#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

static void load_vec(float *i_vec, hls::stream<float> &inStream, int vec_size)
{
mem_rd:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE II = 1
    inStream << i_vec[i];
  }
}

static void compute_rope(hls::stream<float> &q_in_stream,
                         hls::stream<float> &k_in_stream,
                         hls::stream<float> &cos_vec_stream,
                         hls::stream<float> &sin_vec_stream,
                         hls::stream<float> &q_out_stream,
                         hls::stream<float> &k_out_stream, int head_begin)
{
#pragma HLS INLINE off

  float q_local[288];
  float k_local[288];

  float cos_local[24];
  float sin_local[24];

  float q_out_local[288];
  float k_out_local[288];

#pragma HLS ARRAY_PARTITION variable = q_local cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = k_local cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = q_out_local cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = k_out_local cyclic factor = 32

load_qk:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE II = 1
    q_local[i] = q_in_stream.read();
    k_local[i] = k_in_stream.read();
  }

load_cossin:
  for (int i = 0; i < 24; i++)
  {
#pragma HLS PIPELINE II = 1
    cos_local[i] = cos_vec_stream.read();
    sin_local[i] = sin_vec_stream.read();
  }

rotate:
  for (int i = 0; i < 48; ++i)
  {
#pragma HLS PIPELINE II = 1
    int i0 = head_begin + i * 2;
    int i1 = i0 + 1;

    float q0 = q_local[i0];
    float q1 = q_local[i1];

    float k0 = k_local[i0];
    float k1 = k_local[i1];

    float cos = cos_local[i];
    float sin = sin_local[i];

    q_out_local[i0] = q0 * cos - q1 * sin;
    q_out_local[i1] = q0 * sin + q1 * cos;

    k_out_local[i0] = k0 * cos - k1 * sin;
    k_out_local[i1] = k0 * sin + k1 * cos;
  }

store_qk:
  for (int i = 0; i < 288; i++)
  {
#pragma HLS PIPELINE II = 1
    q_out_stream << q_out_local[i];
    k_out_stream << k_out_local[i];
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
  void kernel_rope(float *q_in, float *k_in, float *cos_vec, float *sin_vec,
                   float *q_out, float *k_out, int head_begin)
  {
    #pragma HLS INTERFACE m_axi port = q_in bundle = gmem0 max_widen_bitwidth = 512
    #pragma HLS INTERFACE m_axi port = k_in bundle = gmem1 max_widen_bitwidth = 512
    #pragma HLS INTERFACE m_axi port = cos_vec bundle = gmem2 max_widen_bitwidth = 512
    #pragma HLS INTERFACE m_axi port = sin_vec bundle = gmem3 max_widen_bitwidth = 512
    #pragma HLS INTERFACE m_axi port = q_out bundle = gmem0 max_widen_bitwidth = 512
    #pragma HLS INTERFACE m_axi port = k_out bundle = gmem1 max_widen_bitwidth = 512
    #pragma HLS INTERFACE s_axilite port = q_in bundle = control
    #pragma HLS INTERFACE s_axilite port = k_in bundle = control
    #pragma HLS INTERFACE s_axilite port = cos_vec bundle = control
    #pragma HLS INTERFACE s_axilite port = sin_vec bundle = control
    #pragma HLS INTERFACE s_axilite port = q_out bundle = control
    #pragma HLS INTERFACE s_axilite port = k_out bundle = control
    #pragma HLS INTERFACE s_axilite port = head_begin bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static hls::stream<float> q_in_stream("q_in_stream");
    static hls::stream<float> k_in_stream("k_in_stream");
    static hls::stream<float> cos_vec_stream("cos_vec_stream");
    static hls::stream<float> sin_vec_stream("sin_vec_stream");
    static hls::stream<float> q_out_stream("q_out_stream");
    static hls::stream<float> k_out_stream("k_out_stream");

    #pragma HLS STREAM variable = q_in_stream depth = 288
    #pragma HLS STREAM variable = k_in_stream depth = 288
    #pragma HLS STREAM variable = cos_vec_stream depth = 24
    #pragma HLS STREAM variable = sin_vec_stream depth = 24
    #pragma HLS STREAM variable = q_out_stream depth = 288
    #pragma HLS STREAM variable = k_out_stream depth = 288

    #pragma HLS dataflow
    load_vec(q_in, q_in_stream, 288);
    load_vec(k_in, k_in_stream, 288);
    load_vec(cos_vec, cos_vec_stream, 24);
    load_vec(sin_vec, sin_vec_stream, 24);
    compute_rope(q_in_stream, k_in_stream, cos_vec_stream, sin_vec_stream,
                 q_out_stream, k_out_stream, head_begin);
    store_result(q_out, q_out_stream, 288);
    store_result(k_out, k_out_stream, 288);
  }
}

#endif // USE_CPU_ONLY
