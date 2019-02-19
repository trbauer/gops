#ifndef CLAPP_HPP
#define CLAPP_HPP

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <cl/cl.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <fstream>

namespace mincl
{
  template <typename T,typename...Ts>
  static void concatUnpack(std::stringstream &ss, T t) {
    ss << t;
  }
  template <typename T,typename...Ts>
  static void concatUnpack(std::stringstream &ss, T t, Ts...ts) {
    ss << t;
    concatUnpack(ss, ts...);
  }
  template <typename...Ts>
  static std::string concat(Ts...ts)
  {
    std::stringstream ss;
    concatUnpack(ss,ts...);
    return ss.str();
  }

  template <typename...Ts>
  static void fatal(Ts...ts)
  {
    std::cerr << concat(ts...);
    exit(EXIT_FAILURE);
  }

  #define CL_COMMAND(FUNC,...) \
    do { \
        cl_int __err = FUNC(__VA_ARGS__); \
        if (__err != CL_SUCCESS) \
          mincl::fatal(#FUNC, "near line ", __LINE__, " returned ", __err); \
    } while (0);
  #define CL_COMMAND_CREATE(ASSIGN_TO,FUNC,...) \
    do { \
        cl_int __err = 0; \
        ASSIGN_TO = FUNC(__VA_ARGS__,&__err); \
        if (__err != CL_SUCCESS) \
          mincl::fatal(#FUNC, "near line ", __LINE__, " returned ", __err); \
    } while (0);


  struct ndr {
    size_t dims[3];
    size_t num_dims;
    ndr() : num_dims(0) {dims[0] = dims[0] = dims[2] = 0;}
    ndr(size_t gx)  : num_dims(1) {dims[0] = gx; dims[1] = dims[2] = 1;}
    ndr(size_t gx, size_t gy) 
      : num_dims(2) {dims[0] = gx; dims[1] = gy; dims[2] = 1;}
    ndr(size_t gx, size_t gy, size_t gz)
      : num_dims(3) {dims[0] = gx; dims[1] = gy; dims[2] = gz;}
  };


  void CL_CALLBACK contextCallbackDispatcher(
    const char *, const void *, size_t, void *);

  /////////////////////////////////////////////////////////////////////////////
  // primitive host-side support for half data type
  static float           half_bits_to_float(uint16_t);
  static uint16_t        float_to_half_bits(float);
  //
  struct half {
    uint16_t bits;

    half() {}
    half(float f) : bits(float_to_half_bits(f)) { }
    half(double f) : half((float)f) { }
    half(int64_t i) : half((double)i) { }
    half(uint64_t i) : half((double)i) { }
    operator float() const {return half_bits_to_float(bits);}
    operator double() const {return (float)*this;}

    explicit operator int() const{return (int)half_bits_to_float(bits);}

    // const half &operator=(const half &rhs){bits = rhs.bits; return *this;}

    half operator+(const half &rhs) const {return half(half_bits_to_float(bits) + half_bits_to_float(rhs.bits));}
    half operator-(const half &rhs) const {return half(half_bits_to_float(bits) - half_bits_to_float(rhs.bits));}
    half operator*(const half &rhs) const {return half(half_bits_to_float(bits) * half_bits_to_float(rhs.bits));}
    half operator/(const half &rhs) const {return half(half_bits_to_float(bits) / half_bits_to_float(rhs.bits));}
    bool operator==(const half &rhs) const {return half_bits_to_float(bits) == half_bits_to_float(rhs.bits);}
    bool operator!=(const half &rhs) const {return !(*this == rhs);}
    bool operator<(const half &rhs) const {return half_bits_to_float(bits) < half_bits_to_float(rhs.bits);}
    bool operator<=(const half &rhs) const {return half_bits_to_float(bits) <= half_bits_to_float(rhs.bits);}
    bool operator>(const half &rhs) const {return !(*this <= rhs);}
    bool operator>=(const half &rhs) const {return !(*this < rhs);}
  };
  static_assert(sizeof(half) == sizeof(uint16_t),"wrong size for cls::half");
  static_assert(sizeof(half) == 2,"unexpected size for half");



  class Sample {
    struct BufferBase {
      size_t    length_in_bytes;
      cl_mem    memobj;
      BufferBase(size_t _length_in_bytes, cl_mem _memobj)
        : length_in_bytes(_length_in_bytes), memobj(_memobj) { }
    };
  public:
    template <typename T>
    struct Buffer : BufferBase {
      Buffer(size_t _elems, cl_mem _memobj)
        : BufferBase(_elems*sizeof(T), _memobj)
      {
      }
    };
  private:
    cl_device_id                                   device_id;
    const char                                    *file_name;
    cl_context                                     context;
    cl_command_queue                               queue;
    cl_program                                     program;
    std::vector<std::pair<std::string,cl_kernel>>  kernels;
    std::vector<BufferBase *>                      buffers;

    static std::string loadSource(const char *file_name)
    {
      std::string source;
      std::ifstream ifs(file_name, std::ios::binary);
      if (!ifs.good()) {
        fatal(file_name, ": file not found");
      }
      source.append(
        std::istreambuf_iterator<char>(ifs),
        std::istreambuf_iterator<char>());
      return source;
    }

  public:
    Sample(
      cl_device_id _dev_id,
      const char *_file_name,
      std::string _source,
      const char *_build_options = nullptr,
      cl_command_queue_properties qprops = 0) // CL_QUEUE_PROFILING_ENABLE
      : device_id(_dev_id)
      , file_name(_file_name)
    {
      const char *source_ptr = _source.c_str();
      size_t len = _source.size();
      CL_COMMAND_CREATE(context,
        clCreateContext,
          nullptr, 1, &_dev_id, contextCallbackDispatcher, this);
      CL_COMMAND_CREATE(queue,
        clCreateCommandQueue, context, device_id, qprops);
      //
      // TODO: clCreateProgramWithBinary
      //       clCreateProgramWithIL
      CL_COMMAND_CREATE(program,
        clCreateProgramWithSource,
          context, 1, &source_ptr, &len);
      //
      cl_int err =
        clBuildProgram(program, 1, &device_id, _build_options, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        size_t log_len = 0;
        CL_COMMAND(clGetProgramBuildInfo,
          program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
        char *log = new char[log_len + 1];
        memset(log, 0, log_len+1);
        CL_COMMAND(clGetProgramBuildInfo,
          program, device_id, CL_PROGRAM_BUILD_LOG, log_len, log, nullptr);
        fatal(file_name == nullptr ? "<immediate string>" : file_name,
          ": build failure\n", log);
        delete[] log;
      }
      //
      cl_uint num_kernels = 0;
      CL_COMMAND(clCreateKernelsInProgram,
        program, 0, nullptr, &num_kernels);
      cl_kernel *ks = (cl_kernel *)malloc(num_kernels*sizeof(cl_kernel));
      CL_COMMAND(clCreateKernelsInProgram,
        program, num_kernels, ks, nullptr);
      kernels.reserve(num_kernels);
      for (cl_uint i = 0; i < num_kernels; i++) {
        size_t slen = 0;
        CL_COMMAND(clGetKernelInfo,
          ks[i], CL_KERNEL_FUNCTION_NAME, 0, nullptr, &slen);
        char *knm = (char *)malloc(slen + 1);
        CL_COMMAND(clGetKernelInfo,
          ks[i], CL_KERNEL_FUNCTION_NAME, slen, knm, nullptr);
        knm[slen] = 0;
        kernels.emplace_back(knm, ks[i]);
      }
      free(ks);
    }
    Sample(
      cl_device_id _dev_id,
      const char *_file_name,
      const char *_build_options = nullptr,
      cl_command_queue_properties qprops = 0)
      : Sample(_dev_id, _file_name, loadSource(_file_name), _build_options)
    {
    }
    ~Sample() {
      for (BufferBase *b : buffers) {
        CL_COMMAND(clReleaseMemObject,b->memobj);
        delete b;
      }
      for (auto &k : kernels)
        CL_COMMAND(clReleaseKernel, k.second);
      CL_COMMAND(clReleaseProgram, program);
      CL_COMMAND(clReleaseCommandQueue, queue);
      CL_COMMAND(clReleaseContext, context);
    }
    void saveBinary(std::string file_name) {
      std::string bin_file_name = file_name;
      // Could use .ptx for NVidia
      //
      // auto dot_ix = bin_file_name.rfind('.');
      // if (dot_ix != std::string::npos) {
      //   bin_file_name = bin_file_name.substr(0, dot_ix);
      // }
      // bin_file_name += ".bin";

      size_t bin_size = 0;
      CL_COMMAND(clGetProgramInfo,
        program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr);
      char *bits = new char[bin_size];
      CL_COMMAND(clGetProgramInfo,
        program, CL_PROGRAM_BINARIES, sizeof(bits), &bits, nullptr);

      std::ofstream ofs(bin_file_name, std::ios::binary);
      if (!ofs.good())
        fatal(bin_file_name, ": failed to open output buffer file for writing");
      ofs.write((const char *)bits, bin_size);
      if (!ofs.good())
        fatal(bin_file_name, ": error writing file");

      delete[] bits;
    }

    cl_kernel getKernel(const std::string &kernel_name) {
      for (auto &k : kernels) {
        if (k.first == kernel_name)
          return k.second;
      }
      fatal(kernel_name, ": unable to find kernel");
      return nullptr;
    }

    template <typename E>
    Buffer<E> bufferAllocate(
      size_t length, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
      Buffer<E> *be = new Buffer<E>(length, nullptr);
      CL_COMMAND_CREATE(be->memobj,
        clCreateBuffer,
          context, flags, be->length_in_bytes, nullptr);
      buffers.push_back(be);
      return *be;
    }
    template <typename E>
    Buffer<E> bufferAllocate(
      size_t length, E init, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
      Buffer<E> &be = bufferAllocate<E>(length, flags);
      bufferWrite<E>(be,
         [&] (E *es) {
           for (size_t i = 0; i < length; i++)
             es[i] = init;
         });
      return be;
    }
    template <typename E>
    Buffer<E> bufferAllocate(
      size_t length, const E *init, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
      Buffer<E> &be = bufferAllocate<E>(length, flags);
      bufferWrite<E>(be,
         [&] (E *es) {
           for (size_t i = 0; i < length; i++)
             es[i] = init[i];
         });
      return be;
    }

    template <typename E>
    void bufferRead(
      Buffer<E> b,
      std::function<void (const E *)> apply)
    {
      void *host_ptr = nullptr;
      CL_COMMAND_CREATE(host_ptr,
        clEnqueueMapBuffer,
          queue, b.memobj, CL_TRUE, CL_MAP_READ,
          0, b.length_in_bytes,
          0, nullptr, nullptr);
      apply((const E *)host_ptr);
      CL_COMMAND(clEnqueueUnmapMemObject,
        queue, b.memobj, host_ptr, 0, nullptr, nullptr);
    }
    template <typename E1,typename E2>
    void bufferRead(
      Buffer<E1> b1,
      Buffer<E2> b2,
      std::function<void (const E1 *,const E2 *)> apply)
    {
      bufferRead<E1>(b1, [&] (const E1 *e1) {
        bufferRead<E2>(b2, [&] (const E2 *e2) {
          apply(e1, e2);
        });
      });
    }
    // like bufferRead, but allows a return type
    template <typename E,typename R>
    R bufferReduce(
      Buffer<E> b,
      std::function<R (const E *)> apply)
    {
      const void *host_ptr = nullptr;
      CL_COMMAND_CREATE(host_ptr,
        clEnqueueMapBuffer,
          queue, b.memobj, CL_TRUE, CL_MAP_READ,
          0, b.length_in_bytes,
          0, nullptr, nullptr);
      R ret = apply((E*)host_ptr);
      CL_COMMAND(clEnqueueUnmapMemObject,
        queue, b.memobj, host_ptr, 0, nullptr, nullptr);
      return ret;
    }
    template <typename E>
    void bufferWrite(
      Buffer<E> b,
      std::function<void (E *)> apply)
    {
      void *host_ptr = nullptr;
      CL_COMMAND_CREATE(host_ptr,
        clEnqueueMapBuffer,
          queue, b.memobj, CL_TRUE, CL_MAP_WRITE,
          0, b.length_in_bytes,
          0, nullptr, nullptr);
      apply((E*)host_ptr);
      CL_COMMAND(clEnqueueUnmapMemObject,
        queue, b.memobj, host_ptr, 0, nullptr, nullptr);
    }

    template <typename T>
    void setArgUniform(cl_kernel kernel, cl_uint arg_ix, T t) const {
      CL_COMMAND(clSetKernelArg,kernel, arg_ix, sizeof(T), &t);
    }
    template <typename T>
    void setArgUniform(cl_uint arg_ix, T t) const {
      if (kernels.empty())
        fatal("no kernels in program");
      setArgUniform(kernels.front(), arg_ix, t);
    }
    template <typename T>
    void setArgMem(cl_kernel kernel, cl_uint arg_ix, Buffer<T> &b) const {
      CL_COMMAND(clSetKernelArg,kernel, arg_ix, sizeof(cl_mem), &b.memobj);
    }
    template <typename T>
    void setArgMem(cl_uint arg_ix, Buffer<T> &b) const {
      if (kernels.empty())
        fatal("no kernels in program");
      setArgMem(kernels.front(), arg_ix, b);
    }
    template <typename T>
    void setArgSLM(cl_kernel kernel, cl_uint arg_ix, size_t slm_len) const {
      CL_COMMAND(clSetKernelArg,kernel, arg_ix, sizeof(slm_len), &slm_len);
    }
    template <typename T>
    void setArgSLM(cl_uint arg_ix, size_t slm_len) const {
      if (kernels.empty())
        fatal("no kernels in program");
      setArgSLM(kernels.front(), arg_ix, slm_len);
    }
    void dispatch(cl_kernel k, ndr gsz, ndr lsz, cl_event *evt = nullptr) {
      const size_t *lsz_ptr = lsz.num_dims == 0 ? nullptr : &lsz.dims[0];
      CL_COMMAND(clEnqueueNDRangeKernel,
        queue, k, (cl_uint)gsz.num_dims,
        nullptr, &gsz.dims[0], lsz_ptr,
        0, nullptr, evt);
    }
    void sync() {CL_COMMAND(clFinish, queue);}

    virtual void contextCallback(
      const char * /* errinfo */,
      const void * /* private_info */,
      size_t /* cb */) { }
  }; // class Sample

  static void contextCallbackDispatcher(
    const char *errinfo, const void *private_info, size_t cb, void *env)
  {
    ((Sample *)env)->contextCallback(errinfo, private_info, cb);
  }

  template <typename T> static const char *typeNameToString();
  template <typename T> static const char *elementTypeNameToString();
  template <typename T> static int vectorElements();
#define MAKE_TYPE_NAME_TO_STRING_VEC(T,N)\
  template <> static const char *typeNameToString<cl_ ## T ## N>() {return #T ## #N;}\
  template <> static const char *elementTypeNameToString<cl_ ## T ## N>() {return #T;}\
  template <> static int vectorElements<cl_ ## T ## N>() {return N;}
#define MAKE_TYPE_NAME_TO_STRING(T)\
  template <> static const char *typeNameToString<cl_ ## T>() {return #T;}\
  template <> static const char *elementTypeNameToString<cl_ ## T>() {return #T;}\
  template <> static int vectorElements<cl_ ## T>() {return 1;}\
  MAKE_TYPE_NAME_TO_STRING_VEC(T,2)\
  MAKE_TYPE_NAME_TO_STRING_VEC(T,4)\
  MAKE_TYPE_NAME_TO_STRING_VEC(T,8)\
  MAKE_TYPE_NAME_TO_STRING_VEC(T,16)
  //
  //
  MAKE_TYPE_NAME_TO_STRING(uchar)
  MAKE_TYPE_NAME_TO_STRING(ushort)
  MAKE_TYPE_NAME_TO_STRING(uint)
  MAKE_TYPE_NAME_TO_STRING(ulong)
  //
  MAKE_TYPE_NAME_TO_STRING(char)
  MAKE_TYPE_NAME_TO_STRING(short)
  MAKE_TYPE_NAME_TO_STRING(int)
  MAKE_TYPE_NAME_TO_STRING(long)
  //
  MAKE_TYPE_NAME_TO_STRING(double)
  MAKE_TYPE_NAME_TO_STRING(float)
  //
  // MAKE_TYPE_NAME_TO_STRING(half)
  // cl_half expands to ushort, so we defined clapp::half to be distinct
  // cl_half2 does exist and is a unique type
  template <> static const char *typeNameToString<half>() {return "half";}\
  template <> static const char *elementTypeNameToString<half>() {return "half";}\
  template <> static int vectorElements<half>() {return 1;}\
  MAKE_TYPE_NAME_TO_STRING_VEC(half,2)\
  MAKE_TYPE_NAME_TO_STRING_VEC(half,4)\
  MAKE_TYPE_NAME_TO_STRING_VEC(half,8)\
  MAKE_TYPE_NAME_TO_STRING_VEC(half,16)

#undef MAKE_TYPE_NAME_TO_STRING
#undef MAKE_TYPE_NAME_TO_STRING_VEC

  struct opts {
    int                            verbosity = 0; // -v=...
    std::vector<cl_device_id>      devices; // -d=...
    std::string                    extra_build_options; // -b=... (concatenated)
    std::vector<std::string>       args; // non-options
  };


  static std::vector<std::pair<cl_device_id,std::string>> allDevices()
  {
    std::vector<std::pair<cl_device_id,std::string>> all_ds;

    cl_uint num_ps = 0;
    CL_COMMAND(clGetPlatformIDs, 0, nullptr, &num_ps);
    cl_platform_id *ps = new cl_platform_id[num_ps];
    CL_COMMAND(clGetPlatformIDs, num_ps, ps, nullptr);
    for (cl_uint p_ix = 0; p_ix < num_ps; p_ix++) {
      cl_uint num_ds = 0;
      CL_COMMAND(clGetDeviceIDs, 
        ps[p_ix], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_ds);
      cl_device_id *ds = new cl_device_id[num_ds];
      CL_COMMAND(clGetDeviceIDs, 
        ps[p_ix], CL_DEVICE_TYPE_ALL, num_ds, ds, nullptr);
      for (cl_uint d_ix = 0; d_ix < num_ds; d_ix++) {
        size_t name_len = 0;
        CL_COMMAND(clGetDeviceInfo,
          ds[d_ix],CL_DEVICE_NAME,0,nullptr,&name_len);
        char *namebuf = new char[name_len+1];
        memset(namebuf,0,name_len+1);
        CL_COMMAND(clGetDeviceInfo,
          ds[d_ix],CL_DEVICE_NAME,name_len,namebuf,nullptr);
        std::string dev_name = namebuf;
        all_ds.emplace_back(ds[d_ix], dev_name);
      }
      delete[] ds;
    }
    delete[] ps;

    return all_ds;
  }

  static void listDevices()
  {
    int ix = 0;
    std::cout << "DEVICES:\n";
    for (const auto &p : allDevices()) {
      std::cout << "  #" << ix++ << "   " << p.second << "\n";
    }
  }

  static cl_device_id findDevice(std::string dev_str) {
    int target_dev_ix = -1;
    try {
      target_dev_ix = (int)std::stol(dev_str,nullptr,10);
    } catch (...) {
      // match by name
    }

    cl_device_id picked_device = nullptr;
    std::string picked_device_name;

    auto all_ds = allDevices();
    if (target_dev_ix >= 0) {
      // find by index
      if (target_dev_ix >= (int)all_ds.size()) {
        fatal("device index is out of bounds");
      }
      picked_device = all_ds[target_dev_ix].first;
      picked_device_name = all_ds[target_dev_ix].second;
    } else {
      // find by name
      std::stringstream all_matching_devices;
      int matched = 0;
      for (const auto &d : all_ds) {
        if (d.second.find(dev_str) != std::string::npos) {
          picked_device = d.first;
          picked_device_name = d.second;
          all_matching_devices << " * " << d.second << "\n";
          matched++;
        }
      }
      if (matched == 0) {
        fatal(dev_str, ": unable to match device string");
      } else if (matched > 1) {
        fatal(dev_str, ": is ambiguous amongst:\n", all_matching_devices.str());
      }
    }
    // std::cout << "DEVICE: " << picked_device_name << "\n";
    return picked_device;
  }

  static opts parseOpts(int argc, const char **argv, std::string usage_pfx)
  {
    opts o;
    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
        std::cout <<
          usage_pfx <<
          "where OPTIONS:\n"
          "    -b=OPT        adds an additional build option (e.g. -b=-DTYPE=float2)\n"
          "    -d=DEVICE     sets the device based on index or substring of CL_DEVICE_NAME\n"
          "                  -d=?  lists devices\n"
          "    -q/-v/-v2     sets verbosity (quiet, vebose, debug)\n"
          ;
        exit(EXIT_SUCCESS);
      /////////////////////////////////////////////////////////////////////////
      // VERBOSITY
      } else if (std::string(argv[i]) == "-q") {
        o.verbosity = -1;
      } else if (std::string(argv[i]) == "-v") {
        o.verbosity = 1;
      } else if (std::string(argv[i]) == "-v2") {
        o.verbosity = 2;
      /////////////////////////////////////////////////////////////////////////
      // DEVICE
      } else if (std::string(argv[i]) == "-d") {
        if (i == argc - 1) {
          fatal("-d expects argument");
        }
        std::string val = argv[i+1];
        if (val == "?" || val == "") {
          listDevices();
          exit(EXIT_SUCCESS);
        }
        o.devices.emplace_back(findDevice(val));
        i++;
      } else if (std::string(argv[i]).substr(0,3) == "-d=") {
        std::string val = argv[i] + 3;
        if (val == "?" || val == "") {
          listDevices();
          exit(EXIT_SUCCESS);
        }
        o.devices.emplace_back(findDevice(val));
      /////////////////////////////////////////////////////////////////////////
      // BUILD OPTS
      } else if (std::string(argv[i]) == "-b") {
        if (i == argc - 1) {
          fatal("-b expects argument");
        }
        if (!o.extra_build_options.empty())
          o.extra_build_options += " ";
        o.extra_build_options += argv[i+1];
        i++;
      } else if (std::string(argv[i]).substr(0,3) == "-b=") {
        if (!o.extra_build_options.empty())
          o.extra_build_options += " ";
        o.extra_build_options += argv[i] + 3;
      /////////////////////////////////////////////////////////////////////////
      // ARG or BAD OPT
      } else if (std::string(argv[i]).substr(0,1) == "-") {
        fatal(argv[i],": invalid option");
      } else {
        o.args.push_back(argv[i]);
      }
    }
    return o;
  }

  static std::string getDeviceInfoString(
    cl_device_id dev_id, cl_device_info param_name)
  {
    size_t sz = 0;
    CL_COMMAND(clGetDeviceInfo,
      dev_id, param_name, 0, nullptr, &sz);
    char *sbuf = (char *)alloca(sz+1);
    CL_COMMAND(clGetDeviceInfo,
      dev_id, param_name, sz, sbuf, 0);
    sbuf[sz] = 0;
    return std::string(sbuf);
  }


static const uint32_t F32_SIGN_BIT  = 0x80000000;
static const uint32_t F32_EXP_MASK  = 0x7F800000;
static const uint32_t F32_MANT_MASK = 0x007FFFFF;
static const uint32_t F32_QNAN_BIT  = 0x00400000;
static const int F32_MANTISSA_BITS = 23;
static const uint16_t F16_SIGN_BIT  = 0x8000;
static const uint16_t F16_EXP_MASK  = 0x7C00;
static const uint16_t F16_MANT_MASK = 0x03FF;
static const uint16_t F16_QNAN_BIT  = 0x0200;
static const int F16_MANTISSA_BITS = 10;


  static uint32_t float_to_bits(float f) {
    union{float f; uint32_t i;} u;
    u.f = f;
    return u.i;
  }

  static float float_from_bits(uint32_t f) {
    union{float f; uint32_t i;} u;
    u.i = f;
    return u.f;
  }

  float half_bits_to_float(uint16_t h)
  {
    uint16_t u16 = h;
    static const int MANTISSA_DIFFERENCE = // 23 - 10
       F32_MANTISSA_BITS - F16_MANTISSA_BITS;
    const int F32_F16_BIAS_DIFFERENCE = 127 - 15;

    uint32_t s32 = ((uint32_t)u16 & F16_SIGN_BIT) << 16;
    uint32_t e16 = (u16 & F16_EXP_MASK) >> F16_MANTISSA_BITS;
    uint32_t m16 = u16 & F16_MANT_MASK;

    uint32_t m32, e32;
    if (e16 != 0 && e16 < (F16_EXP_MASK >> F16_MANTISSA_BITS)) { // e16 < 0x1F
      //  normal number
      e32 = e16 + F32_F16_BIAS_DIFFERENCE;
      m32 = m16 << MANTISSA_DIFFERENCE;
    } else if (e16 == 0 && m16 != 0) {
      // denorm/subnorm number (e16 == 0)
      // shift the mantissa left until the hidden one gets set
      for (e32 = (F32_F16_BIAS_DIFFERENCE + 1);
          (m16 & (F16_MANT_MASK + 1)) == 0;
          m16 <<= 1, e32--)
          ;
      m32 = (m16 << MANTISSA_DIFFERENCE) & F32_MANT_MASK;
    } else if (e16 == 0) { // +/- 0.0
      e32 = 0;
      m32 = 0;
    } else {
      e32 = F32_EXP_MASK >> F32_MANTISSA_BITS;
      if (m16 == 0) { // Infinity
        m32 = 0;
      } else { // NaN:  m16 != 0 && e16 == 0x1F
        m32 = (u16 & F16_QNAN_BIT) << MANTISSA_DIFFERENCE; // preserve sNaN bit
        m32 |= (F16_MANT_MASK >> 1) & m16;
        if (m32 == 0) {
            m32 = 1; // ensure still NaN
        }
      }
    }
    return float_from_bits(s32 | (e32 << F32_MANTISSA_BITS) | m32);
  }

  uint16_t float_to_half_bits(float f)
  {
    uint32_t f32 = float_to_bits(f);

    uint32_t m32 = F32_MANT_MASK & f32;
    uint32_t e32 = (F32_EXP_MASK & f32) >> F32_MANTISSA_BITS;

    uint32_t m16;
    uint32_t e16;

    if (e32 == (F32_EXP_MASK >> F32_MANTISSA_BITS)) {
      // NaN or Infinity
      e16 = F16_EXP_MASK;
      m16 = (F16_MANT_MASK >> 1) & f32;
      if (m32 != 0) {
        // preserve the bottom 9 bits of the NaN payload and
        // shift the signaling bit (high bit) down as bit 10
        m16 |= (F32_QNAN_BIT & f32) >>
            (F32_MANTISSA_BITS - F16_MANTISSA_BITS);
        // s eeeeeeee mmmmmmmmmmmmmmmmmmmmmm
        //            |            |||||||||
        //            |            vvvvvvvvv
        //            +---------->mmmmmmmmmm
        if (m16 == 0) {
            // if the nonzero payload is in the high bits and and gets
            // dropped and the signal bit is non-zero, then m16 is 0,
            // to maintain it as a qnan, we must set at least one bit
            m16 = 0x1;
        }
      }
    } else if (e32 > (127 - 15) + 0x1E) { // e16 overflows 5 bits after bias fix
      // Too large for f16 => infinity
      e16 = F16_EXP_MASK;
      m16 = 0;
    } else if (e32 <= (127 - 15) && e32 >= 0x66) {
      // Denorm/subnorm float
      //
      // Normal floats are:
      //   (1 + sum{m[i]^(23-i)*2^(-i)}) * 2^(e - bias)
      //   (each mantissa bit is a fractional power of 2)
      // Denorms are:
      //   (0 + ...)
      // This is a zero exponent, but non-zero mantissa
      //
      // set leading bit past leading mantissa bit (low exponent bit)
      // (hidden one)
      m32 |= (F32_QNAN_BIT << 1);
      // exponent
      // repeatedly increment the f32 exponent and divide the denorm
      // mantissa until the exponent reachs a non-zero value
      for (; e32 <= 127 - 15; m32 >>= 1, e32++)
          ;
      e16 = 0;
      m16 = m32 >> (F32_MANTISSA_BITS - F16_MANTISSA_BITS);
    } else if (e32 < 0x66) {
      // Too small: rounds to +/-0.0
      e16 = 0;
      m16 = 0;
    } else {
      // Normalized float
      e16 = (e32 - (127 - 15)) << F16_MANTISSA_BITS;
      m16 = m32 >> (F32_MANTISSA_BITS - F16_MANTISSA_BITS);
      // TODO: rounding modes?
      // if (m32 & 0x1000)
      //   h16++;
      // c.f. https://gist.github.com/rygorous/2156668
      // if (((m32 & 0x1fff) > 0x1000) || (m16 & 1)) // above halfway point or unrounded result is odd
      //   h16++;
    }

    uint32_t s16 = (f32 >> 16) & F16_SIGN_BIT;
    uint16_t h{(uint16_t)(s16 | e16 | m16)};

    return h;
  }


} //  mincl namespace
#endif