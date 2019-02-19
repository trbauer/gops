#include "mincl.hpp"

#include <cstdint>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


static const uint64_t TOTAL_WORKITEMS = 1024*1024; // SIMD32 => 32k threads
//
static const uint64_t LOOP_TRIPS = 4;
static const uint64_t OPS_PER_LOOP = 256; // ~4k given 16B instructions
//
static const uint64_t OPS_PER_WORKITEM = LOOP_TRIPS*OPS_PER_LOOP; // 1k

// enables ILP (only every fourth instruction depends on it's last result)
static const int PARALLEL_SUMS = 4;

template <typename T>
static double computeGigaOpsPerSecond(
  mincl::opts os,
  cl_device_id dev_id,
  std::stringstream &verbose_output)
{
  const int NUM_INPUTS = 4;
  std::string type = mincl::typeNameToString<T>();

  std::string kernel_name = "ops_"; kernel_name += type;
  std::stringstream ss;
  if (type.find("half") != std::string::npos) {
    ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  } else if (type.find("double") != std::string::npos) {
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  ss <<
    "\n"
    "__kernel void " << kernel_name << "("
    "  __global " << type << "*dst" <<
    "  )\n" <<
    "{\n";
  for (int i = 0; i < NUM_INPUTS; i++) {
    ss <<
      "  " << type << " arg" << i << " = (" << type << ")(\n";
    for (int v = 0; v < mincl::vectorElements<T>(); v++) {
      ss <<
        "    (" << mincl::elementTypeNameToString<T>() <<
        ")(get_local_id(0)+" << (i+1)*(v+1) << ")";
      if (v < mincl::vectorElements<T>() - 1)
        ss << ",\n";
    }
    ss << ");\n";
  }

  for (int i = 0; i < PARALLEL_SUMS; i++) {
    ss <<  "  " << type << " sum" << i << " = arg" << (i % NUM_INPUTS) << ";\n";
  }
  ss << "\n";
  ss << "  for (int l_ix = 0; l_ix < " << LOOP_TRIPS << "; l_ix++) {\n";
  for (int i = 0; i < OPS_PER_LOOP; i++) {
    ss << "    sum" << (i % PARALLEL_SUMS) <<
      " += arg" << (i % NUM_INPUTS) << "*sum" << (i % PARALLEL_SUMS) << ";\n";
  }
  ss << "  }\n";
  ss << "\n";
  ss <<
    "  dst[get_global_id(0)] = ";
  for (int i = 0; i < PARALLEL_SUMS; i++) {
    if (i > 0)
      ss << " + ";
    ss << "sum" << i;
  }
  ss << ";\n";
  ss <<
    "}\n";

  // std::cout << "\n" << ss.str() << "\n";
  std::string build_opts = "-cl-fast-relaxed-math";
  if (!os.extra_build_options.empty()) {
    build_opts += " ";
    build_opts += os.extra_build_options;
  }

  if (os.verbosity > 1) {
    std::ofstream ofs(kernel_name + ".cl");
    ofs << "// with build options " << build_opts << "\n";
    ofs << ss.str();
    ofs.flush();
  }

  // dump the source
  // std::cout << ss.str();

  mincl::Sample s(
    dev_id,
    nullptr,
    ss.str(),
    build_opts.c_str(),
    CL_QUEUE_PROFILING_ENABLE);
  if (os.verbosity > 1)
    s.saveBinary(kernel_name + ".bin");
  //
  auto k = s.getKernel(kernel_name);
  auto b = s.bufferAllocate<T>(TOTAL_WORKITEMS);
  s.setArgMem(k, 0, b);

  ///////////////////////
  // one warmup dispatch
  s.dispatch(k, mincl::ndr(TOTAL_WORKITEMS), mincl::ndr(), nullptr);
  s.sync();

  ///////////////////////
  // time it
  cl_event evt;
  s.dispatch(k, mincl::ndr(TOTAL_WORKITEMS), mincl::ndr(), &evt);
  s.sync();

  cl_ulong ns_start = 0;
  CL_COMMAND(clGetEventProfilingInfo,
    evt, CL_PROFILING_COMMAND_START, sizeof(ns_start), &ns_start, nullptr);
  cl_ulong ns_end = 0;
  CL_COMMAND(clGetEventProfilingInfo,
    evt, CL_PROFILING_COMMAND_END, sizeof(ns_end), &ns_end, nullptr);
  double s_elapsed = (ns_end - ns_start)/1000.0/1000.0/1000.0;
  CL_COMMAND(clReleaseEvent, evt);

  if (os.verbosity > 0) {
    if (mincl::vectorElements<T>() > 1) {
      verbose_output << mincl::vectorElements<T>() << " (vector) * ";
    }
    verbose_output << TOTAL_WORKITEMS*OPS_PER_WORKITEM << " ops in " <<
      std::fixed << std::setprecision(3) << s_elapsed << " s";
  }

  return mincl::vectorElements<T>()*
    ((TOTAL_WORKITEMS*OPS_PER_WORKITEM)/s_elapsed)/1000.0/1000.0/1000.0; // GOPS
}

///////////////////////////////////////////////////////////////////////////////
const static int TYPE_NAME_COLUMN_WIDTH = 16;
const static int DEVICE_NAME_COLUMN_WIDTH = 32;

template <typename T>
static void testTypeOnDevices(mincl::opts os)
{
  std::string type_name = mincl::typeNameToString<T>();
  std::cout << std::setw(TYPE_NAME_COLUMN_WIDTH) <<
    mincl::typeNameToString<T>();

  std::stringstream verbose_outputs;

  for (const auto &dev : os.devices) {
    std::stringstream verbose_output;
    //
    std::string exts = mincl::getDeviceInfoString(dev, CL_DEVICE_EXTENSIONS);
    //
    bool test_is_fp16 = type_name.find("half") != std::string::npos;
    bool dev_supports_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
    bool test_is_fp64 = type_name.find("double") != std::string::npos;
    bool dev_supports_fp64 = exts.find("cl_khr_fp16") != std::string::npos;
    //
    std::cout << "  ";
    double gops;
    if (test_is_fp16 && !dev_supports_fp16) {
      std::cout << std::setw(DEVICE_NAME_COLUMN_WIDTH) << "no cl_khr_fp16";
    } else if (test_is_fp64 && !dev_supports_fp64) {
      std::cout << std::setw(DEVICE_NAME_COLUMN_WIDTH) << "no cl_khr_fp64";
    } else {
      gops = computeGigaOpsPerSecond<T>(os, dev, verbose_output);
      std::cout << std::setw(DEVICE_NAME_COLUMN_WIDTH) <<
        std::fixed << std::setprecision(2) << gops;
    }

    if (verbose_output.tellp() > 0) {
      verbose_outputs <<
        "[" << mincl::getDeviceInfoString(dev, CL_DEVICE_NAME) << "] ";
      auto str = verbose_output.str();
      verbose_outputs << str;
      if (str[str.size() - 1] != '\n' && str[str.size() - 1] != '\r')
        verbose_outputs << "\n";
    }
  }
  std::cout << "  GOPS\n";
  std::cout << verbose_outputs.str();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
  std::stringstream ss;
  ss <<
    "CL Ops (" GOPS_VERSION_STRING ")\n"
    "usage: clops" << GOPS_HOST_POINTER_SIZE;
  const mincl::opts os = mincl::parseOpts(argc, argv, ss.str());
  if (os.devices.empty()) {
  }
  //
  std::cout << std::setw(TYPE_NAME_COLUMN_WIDTH) << "TYPE";
  for (const auto &dev : os.devices) {
    std::cout << "  " << std::setw(DEVICE_NAME_COLUMN_WIDTH) <<
      mincl::getDeviceInfoString(dev, CL_DEVICE_NAME);
  }
  std::cout << "\n";
  //
  ///////////////////////////////////////////
  //        D1        D2
  // T1     ......    .......
  // T2     ......    .......
  // T3     ......    .......
  // T4     ......    .......
  // T5     ......    .......
  testTypeOnDevices<mincl::half>(os); // cl_half typedefs to ushort
  testTypeOnDevices<cl_float>(os);
  testTypeOnDevices<cl_double>(os);
  //
  std::cout << "\n";
  //
  testTypeOnDevices<cl_uchar>(os);
  testTypeOnDevices<cl_ushort>(os);
  testTypeOnDevices<cl_uint>(os);
  testTypeOnDevices<cl_ulong>(os);
  //
  testTypeOnDevices<cl_uchar4>(os);
  testTypeOnDevices<cl_float4>(os);
  // testTypeOnDevices<cl_half2>(os);

  return EXIT_SUCCESS;
}
