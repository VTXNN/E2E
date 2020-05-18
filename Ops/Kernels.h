namespace kernels
{
    enum class KernelType
    {
        Flat = 0,
        Triangle = 1,
    };
    
    inline float clamp(float v, float l, float h)
    {
        return v < l ? l : (v > h ? h : v);
    }
    
    struct FlatKernel
    {
        static inline float integral(float x0, float start, float end, float b)
        {
            //kernel: 1/(2*b) if |x-x0|<b else 0
            //integral: (x/b+1)/2 for x clamped to [-b,b]

            const float cstart = clamp(start-x0,-b,b);
            const float cend = clamp(end-x0,-b,b);
            
            return 0.5*(cend-cstart)/b;
        }
        
        static inline float gradX0(float x0, float start, float end, float b)
        {
            //kernel: 1/(2*b) if |x-x0|<b else 0
            
            const float inv2b = 0.5f/b;
            return (std::fabs(start-x0)<b ? inv2b : 0) - (std::fabs(end-x0)<b ? inv2b : 0);
        }
        
    };
    
    struct TriangleKernel
    {
        static inline float integral(float x0, float start, float end, float b)
        {
            //kernel: 1/b*max(0.,1-|(x-x0)/b|)
            //integral: 0.5-0.5*(x*|x|-2*x*b)/b/b for x clamped to [-b,b]

            const float cstart = clamp(start-x0,-b,b);
            const float cend = clamp(end-x0,-b,b);
            
            const float invb2 = 1.f/(b*b);
            const float intStart = 0.5-0.5*(cstart*std::fabs(cstart)-2*cstart*b)*invb2;
            const float intEnd = 0.5-0.5*(cend*std::fabs(cend)-2*cend*b)*invb2;
            
            return intEnd-intStart;
        }
        
        static inline float gradX0(float x0, float start, float end, float b)
        {
            //kernel: 1/b*max(0.,1-|(x-x0)/b|)
            const float invb = 1.f/b;
            return (std::max(0.f,1-std::fabs((start-x0)*invb))-std::max(0.f,1-std::fabs((end-x0)*invb)))*invb;
        }
    };
}
