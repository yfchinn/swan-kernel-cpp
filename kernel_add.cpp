#ifndef USE_CPU_ONLY

#include <hls_stream.h>
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

static void compute_add(hls::stream<float> &in1_stream,
                        hls::stream<float> &in2_stream,
                        hls::stream<float> &out_stream, int vec_size)
{
compute_loop:
  for (int i = 0; i < vec_size; i++)
  {
#pragma HLS PIPELINE
    float a = in1_stream.read();
    float b = in2_stream.read();
    out_stream << a + b;
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
  void kernel_add(float *i_vec_1, float *i_vec_2, float *o_vec, int vec_size)
  {
// to DRAM
#pragma HLS INTERFACE m_axi port = i_vec_1 bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = i_vec_2 bundle = gmem1 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0 max_write_burst_length = 256
// to PS
#pragma HLS INTERFACE s_axilite port = i_vec_1 bundle = control
#pragma HLS INTERFACE s_axilite port = i_vec_2 bundle = control
#pragma HLS INTERFACE s_axilite port = o_vec bundle = control
#pragma HLS INTERFACE s_axilite port = vec_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static hls::stream<float> vec_stream_1("vec_stream_1");
#pragma HLS STREAM variable = vec_stream_1 depth = 512
    static hls::stream<float> vec_stream_2("vec_stream_2");
#pragma HLS STREAM variable = vec_stream_2 depth = 512
    static hls::stream<float> out_stream("out_stream");
#pragma HLS STREAM variable = out_stream depth = 512

#pragma HLS dataflow
    load_vec(i_vec_1, vec_stream_1, vec_size);
    load_vec(i_vec_2, vec_stream_2, vec_size);
    compute_add(vec_stream_1, vec_stream_2, out_stream, vec_size);
    store_result(o_vec, out_stream, vec_size);
  }
}

#endif // USE_CPU_ONLY
