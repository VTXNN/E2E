#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "Kernels.h"

#include <iostream>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("HistogramMaxSample")
    .Input("hists: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) 
    {
        ShapeHandle values_input_shape = c->input(0);

        ShapeHandle ouput_shape = c->MakeShape({
            c->Dim(values_input_shape,0), //batch
            c->Dim(values_input_shape,2) //hists
        
        });

        c->set_output(0,ouput_shape);

        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");

#include <random>

namespace 
{
    static thread_local std::unique_ptr<std::mt19937> generator_;
}


class HistogramMaxSampleOp:
    public OpKernel
{  
    public:
        explicit HistogramMaxSampleOp(OpKernelConstruction* context):
            OpKernel(context)
        {
        }

        virtual ~HistogramMaxSampleOp()
        {
        }

        void Compute(OpKernelContext* context)
        { 
        
            if (not generator_)
            {
                generator_.reset(new std::mt19937(123456));
            }
        
            const Tensor& values_input_tensor = context->input(0);

            TensorShape outputShape({
                values_input_tensor.dim_size(0),
                values_input_tensor.dim_size(2)
            });

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, outputShape, &output_tensor)
            );
            
            const auto inputs = values_input_tensor.tensor<float,3>();
            auto outputs = output_tensor->tensor<float,2>();
            
            for (int ibatch = 0; ibatch<values_input_tensor.dim_size(0); ++ibatch)
            {   
                for (int ihist = 0; ihist<values_input_tensor.dim_size(2); ++ihist)
                {
                    float sum = 0;
                    for (int ivalue = 0; ivalue<values_input_tensor.dim_size(1); ++ivalue)
                    {
                        sum += std::fabs(inputs(ibatch,ivalue,ihist));
                    }
                    std::uniform_real_distribution<float> dist(0,sum);
                    float value = dist(*generator_);
                    outputs(ibatch,ihist) = 0;
                    for (int ivalue = 0; ivalue<values_input_tensor.dim_size(1); ++ivalue)
                    {
                        sum -= std::fabs(inputs(ibatch,ivalue,ihist));
                        if (sum<value)
                        {
                            outputs(ibatch,ihist) = ivalue;
                            break;
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("HistogramMaxSample").Device(DEVICE_CPU),HistogramMaxSampleOp)

