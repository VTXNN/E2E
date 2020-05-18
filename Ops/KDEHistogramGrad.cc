#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "Kernels.h"

#include <iostream>

using namespace tensorflow;
using namespace tensorflow::shape_inference;
using namespace kernels;

REGISTER_OP("KDEHistogramGrad")
    .Attr("nbins: int >=1")
    .Attr("start: float")
    .Attr("end: float")
    .Attr("kernel: {'flat','triangle'}")
    .Attr("bandwidth_grad: float = 0")
    .Attr("add_overflow: bool")
    .Input("values: float")
    .Input("weights: float")
    .Input("gradients: float")
    .Output("gradients_weights: float")
    .Output("gradients_values: float")
    .SetShapeFn([](InferenceContext* c)
    {
        c->set_output(0,c->input(0));
        c->set_output(1,c->input(1));

        return Status::OK();
    })
    .Doc(R"doc(Produces a 1D Histogram using the given KDE from the input values and weights)doc");


class KDEHistogramGradOp:
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
        explicit KDEHistogramGradOp(OpKernelConstruction* context):
            OpKernel(context)
        {
            OP_REQUIRES_OK(context,context->GetAttr("start",&start));
            OP_REQUIRES_OK(context,context->GetAttr("end",&end));
            OP_REQUIRES_OK(context,context->GetAttr("nbins",&nBins));
            OP_REQUIRES_OK(context,context->GetAttr("bandwidth_grad",&bandWidth));
            OP_REQUIRES_OK(context,context->GetAttr("add_overflow",&addOverflow));

            OP_REQUIRES(
                context,
                (bandWidth>0),
                errors::InvalidArgument("Bandwidth '"+std::to_string(bandWidth)+"' required to be >0")
            );

            OP_REQUIRES(
                context,
                (start<end),
                errors::InvalidArgument("start < end (= "+std::to_string(start)+" < "+std::to_string(end)+") required")
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

        virtual ~KDEHistogramGradOp()
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
            const Tensor& input_values_tensor = context->input(0);
            const Tensor& input_weights_tensor = context->input(1);
            const Tensor& input_gradients_tensor = context->input(2);

            const int batchSize = input_values_tensor.dim_size(0);
            const int nValues = input_values_tensor.dim_size(1);

            Tensor* output_gradients_values = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, input_values_tensor.shape(), &output_gradients_values)
            );

            Tensor* output_gradients_weights = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(1, input_weights_tensor.shape(), &output_gradients_weights)
            );

            const auto values = input_values_tensor.matrix<float>();
            const auto weights = input_weights_tensor.matrix<float>();
            const auto gradients = input_gradients_tensor.matrix<float>();

            auto gradient_values = output_gradients_values->matrix<float>();
            auto gradient_weights = output_gradients_weights->matrix<float>();

            for (int ibatch = 0; ibatch < batchSize; ++ibatch)
            {
                for (int ivalue = 0; ivalue < nValues; ++ivalue)
                {
                    gradient_values(ibatch,ivalue) = 0;
                    gradient_weights(ibatch,ivalue) = 0;
                    
                    const float valueClamped = addOverflow ? clamp(
                        values(ibatch,ivalue),
                        lowerBinEdges[0],
                        lowerBinEdges[nBins]
                    ) : values(ibatch,ivalue);

                    for (int ibin = 0; ibin < nBins; ++ibin)
                    {
                        gradient_values(ibatch,ivalue)+=gradients(ibatch,ibin)*weights(ibatch,ivalue)*KERNELFCT::gradX0(
                            valueClamped,
                            lowerBinEdges[ibin],
                            lowerBinEdges[ibin+1],
                            bandWidth
                        );

                        gradient_weights(ibatch,ivalue)+=gradients(ibatch,ibin)*KERNELFCT::integral(
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

REGISTER_KERNEL_BUILDER(Name("KDEHistogramGrad").Device(DEVICE_CPU),KDEHistogramGradOp)

