#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "Kernels.h"

#include <iostream>

using namespace tensorflow;
using namespace tensorflow::shape_inference;
using namespace kernels;

REGISTER_OP("KDEHistogram")
    .Attr("nbins: int >=1")
    .Attr("start: float")
    .Attr("end: float")
    .Attr("kernel: {'flat','triangle'}")
    .Attr("bandwidth_hist: float")
    .Attr("bandwidth_grad: float") //required here to register gradient later
    .Attr("add_overflow: bool")
    .Input("values: float")
    .Input("weights: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) 
    {
        ShapeHandle values_input_shape = c->input(0);

        int nBins = 0;
        c->GetAttr("nbins",&nBins);

        ShapeHandle ouput_shape = c->MakeShape({
            c->Dim(values_input_shape,0),
            c->MakeDim(nBins)
        });

        c->set_output(0,ouput_shape);

        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");

class KDEHistogramOp:
    public OpKernel
{
    private:
        float start;
        float end;
        int nBins;
        float bandWidth;
        bool addOverflow;

        float* lowerBinEdges;

        KernelType kernelType;

    public:
        explicit KDEHistogramOp(OpKernelConstruction* context):
            OpKernel(context)
        {
            OP_REQUIRES_OK(context,context->GetAttr("start",&start));
            OP_REQUIRES_OK(context,context->GetAttr("end",&end));
            OP_REQUIRES_OK(context,context->GetAttr("nbins",&nBins));
            OP_REQUIRES_OK(context,context->GetAttr("bandwidth_hist",&bandWidth));
            OP_REQUIRES_OK(context,context->GetAttr("add_overflow",&addOverflow));

            OP_REQUIRES(
                context,
                (bandWidth>0),
                errors::InvalidArgument("Bandwidth required to be >0")
            );
            
            OP_REQUIRES(
                context,
                (start<end),
                errors::InvalidArgument("start<end required")
            );

            OP_REQUIRES(
                context,
                (nBins>0),
                errors::InvalidArgument("Number of bins need to be >0")
            );

            std::string kernelName;
            OP_REQUIRES_OK(context,context->GetAttr("kernel",&kernelName));

            if (kernelName=="flat")
            {
                kernelType = KernelType::Flat;
            }
            else if (kernelName=="triangle")
            {
                kernelType = KernelType::Triangle;
            }
            else
            {
                kernelType = KernelType::Flat;
            }

            lowerBinEdges = new float[nBins+1];
            for (int ibin = 0; ibin < (nBins+1); ++ibin)
            {
                lowerBinEdges[ibin] = start+ibin*(end-start)/nBins;
            }
        }

        virtual ~KDEHistogramOp()
        {
        }

        void Compute(OpKernelContext* context)
        {
            switch (kernelType)
            {
                case KernelType::Flat:
                    ComputeTmpl<FlatKernel>(context);
                    break;
                case KernelType::Triangle:
                    ComputeTmpl<TriangleKernel>(context);
                    break;
            }
        }

        template<typename KERNELFCT>
        void ComputeTmpl(OpKernelContext* context)
        {
            const Tensor& values_input_tensor = context->input(0);
            const Tensor& weights_input_tensor = context->input(1);

            const int batchSize = values_input_tensor.dim_size(0);
            const int nValues = values_input_tensor.dim_size(1);

            TensorShape outputShape({
                batchSize,
                nBins
            });

            Tensor* output_histogram_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, outputShape, &output_histogram_tensor)
            );

            const auto values = values_input_tensor.matrix<float>();
            const auto weights = weights_input_tensor.matrix<float>();
            auto output = output_histogram_tensor->matrix<float>();

            for (int ibatch = 0; ibatch < batchSize; ++ibatch)
            {
                for (int ivalue = 0; ivalue < nValues; ++ivalue)
                {
                    const float valueClamped = addOverflow ? clamp(
                        values(ibatch,ivalue),
                        lowerBinEdges[0],
                        lowerBinEdges[nBins]
                    ) : values(ibatch,ivalue);

                    for (int ibin = 0; ibin < nBins; ++ibin)
                    {
                        if (ivalue==0) output(ibatch,ibin) = 0;

                        output(ibatch,ibin)+=weights(ibatch,ivalue)*KERNELFCT::integral(
                            valueClamped,
                            lowerBinEdges[ibin],
                            lowerBinEdges[ibin+1],
                            bandWidth
                        );
                        
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("KDEHistogram").Device(DEVICE_CPU),KDEHistogramOp)

